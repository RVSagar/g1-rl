import os
import functools
import json
import pickle
import argparse
from datetime import datetime

import jax
import jax.numpy as jp
jax.config.update('jax_default_matmul_precision', 'highest')

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import imageio.v3 as iio

from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from mujoco_playground import locomotion, wrapper
from mujoco_playground._src.gait import draw_joystick_command
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


def run_train_and_evaluate(env_name="G1JoystickFlatTerrain", num_timesteps=10_000, render_every=2, save_gif=False):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"

    env_cfg = locomotion.get_default_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_params.num_timesteps = num_timesteps

    env_cfg.reward_config.scales.energy = 0.0
    env_cfg.reward_config.scales.dof_acc = 0.0

    randomizer = locomotion.get_domain_randomizer(env_name)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"
    ckpt_path = epath.Path("checkpoints").resolve() / exp_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {exp_name}")

    with open(ckpt_path / "config.json", "w") as f:
        json.dump(env_cfg.to_dict(), f, indent=2)

    x_data, y_data, y_dataerr = [], [], []
    ep_len_data, ep_len_err = [], []
    times = [datetime.now()]

    def progress_fn(step, metrics):
        times.append(datetime.now())
        reward = float(metrics.get("eval/episode_reward", 0))
        reward_std = float(metrics.get("eval/episode_reward_std", 0))
        ep_len = float(metrics.get("eval/avg_episode_length", 0))
        ep_len_std = float(metrics.get("eval/avg_episode_length_std", 0))

        x_data.append(step)
        y_data.append(reward)
        y_dataerr.append(reward_std)
        ep_len_data.append(ep_len)
        ep_len_err.append(ep_len_std)

        metrics_path = ckpt_path / f"{step}" / "metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({
                "step": step,
                "reward": reward,
                "reward_std": reward_std,
                "ep_length": ep_len,
                "ep_length_std": ep_len_std
            }, f, indent=2)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
        axs[0].set_title(f"Reward: {reward:.2f}")
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Reward")

        axs[1].errorbar(x_data, ep_len_data, yerr=ep_len_err, color="blue")
        axs[1].set_title(f"Episode Length: {ep_len:.2f}")
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Length")

        plt.tight_layout()
        plt.savefig(ckpt_path / "progress.png")
        plt.close(fig)

    def policy_params_fn(step, make_policy, params):
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / str(step)
        checkpointer.save(path, params, force=True, save_args=save_args)

    train_fn = functools.partial(
        ppo.train,
        **{k: v for k, v in ppo_params.items() if k != "network_factory"},
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **ppo_params.network_factory),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        restore_checkpoint_path=None,
        randomization_fn=randomizer,
    )

    env = locomotion.load(env_name, config=env_cfg)
    eval_env = locomotion.load(env_name, config=env_cfg)  # temporary

    #Start timing
    train_start = datetime.now()
    make_inference_fn, params, _ = train_fn(environment=env, eval_env=eval_env)
    train_duration = datetime.now() - train_start

    with open(ckpt_path / "params.pkl", "wb") as f:
        pickle.dump({
            "normalizer_params": params[0],
            "policy_params": params[1],
            "value_params": params[2],
        }, f)

    print(f"Training complete. Checkpoint saved at {ckpt_path}")
    print(f"Training time: {train_duration.seconds // 60} min {train_duration.seconds % 60} sec")

    # === Reload eval env to avoid tracer leaks ===
    print("Evaluating latest policy from memory...")
    eval_env = locomotion.load(env_name, config=locomotion.get_default_config(env_name))

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
    jit_step = jax.jit(eval_env.step)

    rng = jax.random.PRNGKey(1)
    command = jp.array([0.0, 0.0, 0.0])
    phase_dt = 2 * jp.pi * eval_env.dt * 1.5
    phase = jp.array([0, jp.pi])

    state = eval_env.reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase

    rollout, mod_fns = [], []
    for _ in range(eval_env._config.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        if state.done:
            print("Early termination.")
            break
        state.info["command"] = command

        xyz = np.array(state.data.xpos[eval_env.mj_model.body("torso_link").id])
        x_axis = state.data.xmat[eval_env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])

        mod_fns.append(functools.partial(draw_joystick_command, cmd=command, xyz=xyz, theta=yaw, scl=np.linalg.norm(command)))
        rollout.append(state)

    traj = rollout[::render_every]
    mod_fns = mod_fns[::render_every]
    fps = 1.0 / eval_env.dt / render_every

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=1280,
        height=480,
        modify_scene_fns=mod_fns,
    )

    media.show_video(frames, fps=fps, loop=False)

    if save_gif:
        gif_path = ckpt_path / "rollout.gif"
        iio.imwrite(str(gif_path), np.array(frames), fps=fps)
        print(f"Saved GIF to: {gif_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true", help="Save rollout as GIF.")
    parser.add_argument("--timesteps", type=int, default=100_000_000, help="Number of PPO training steps.")
    parser.add_argument("--env_name", type=str, default="G1JoystickFlatTerrain", help="Environment name.")
    args = parser.parse_args()

    run_train_and_evaluate(
        env_name=args.env_name,
        num_timesteps=args.timesteps,
        render_every=2,
        save_gif=args.gif
    )
