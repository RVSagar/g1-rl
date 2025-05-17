"""
Custom configuration for Unitree G1 robot locomotion training.
This is a full copy of the G1 configurations with all parameters exposed for customization.
"""

from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseEnvCfg, BaseAgentCfg, BaseSceneCfg, RobotCfg, DomainRandCfg,
    RewardCfg, HeightScannerCfg, PhysxCfg, SimCfg
)
from legged_lab.assets.unitree import G1_CFG
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG
from isaaclab.managers import RewardTermCfg as RewTerm
import legged_lab.mdp as mdp
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

# Import custom reward functions
import custom_rewards

@configclass
class CustomG1RewardCfg(RewardCfg):
    # Copy all rewards from G1RewardCfg
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5}
    )
    
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"std": 0.5}
    )
    
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0
    )
    
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05
    )
    
    energy = RewTerm(
        func=mdp.energy,
        weight=-1e-3
    )
    
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5e-7
    )
    
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01
    )
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle.*).*"),
            "threshold": 1.0
        }
    )
    
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 1.0
        }
    )
    
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")},
        weight=-2.0
    )
    
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0
    )
    
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 0.4
        }
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*")
        }
    )
    
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 300,
            "max_reward": 400
        }
    )
    
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]),
            "threshold": 0.2
        }
    )
    
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])
        }
    )
    
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0
    )
    
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_yaw.*",
                ".*_hip_roll.*",
                ".*_shoulder_pitch.*",
                ".*_elbow.*"
            ])
        }
    )
    
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*waist.*",
                ".*_shoulder_roll.*",
                ".*_shoulder_yaw.*",
                ".*_wrist.*"
            ])
        }
    )
    
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_hip_pitch.*",
                ".*_knee.*",
                ".*_ankle.*"
            ])
        }
    )
    
    # Add knee behavior rewards
    stance_straight_knee = RewTerm(
        func=custom_rewards.stance_straight_knee,
        weight=-0.1,  # Negative weight since it's a penalty for bent knees
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee.*"])
        }
    )
    
    swing_knee_flexion = RewTerm(
        func=custom_rewards.swing_knee_flexion,
        weight=0.1,  # Positive weight to encourage proper flexion
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee.*"]),
            "target_angle": 0.35  # Target flexion angle in radians
        }
    )

    # heel_strike_with_vel = RewTerm(
    #     func=custom_rewards.heel_strike_with_vel,
    #     weight=0.2,  # Increased weight to emphasize heel strike
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #         "ankle_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch.*"]),
    #         "velocity_threshold": -0.2,     # Negative for downward velocity
    #         "pitch_threshold": 0.05         # Increased threshold for stronger heel-first preference
    #     }
    # )

    # toe_off_with_vel = RewTerm(
    #     func=custom_rewards.toe_off_with_vel,
    #     weight=0.05,  # Reduced weight to make toe-off secondary to heel strike
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    #         "ankle_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_pitch.*"]),
    #         "velocity_threshold": 0.2,      # Keep same upward velocity requirement
    #         "pitch_threshold": -0.025       # Keep same toe-off angle threshold
    #     }
    # )

    smooth_foot_landing = RewTerm(
        func=custom_rewards.smooth_foot_landing,
        weight=-0.1,                       # Increased penalty for hard impacts
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "velocity_threshold": 0.2       # Slightly reduced threshold to enforce softer landing
        }
    )


@configclass
class CustomG1FlatEnvCfg(BaseEnvCfg):
    """Custom G1 flat environment configuration with all parameters exposed."""

    reward = CustomG1RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        # Copy settings from G1FlatEnvCfg
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = [".*torso.*"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*torso.*"]
        
        # Customize any parameters as needed
        # For example:
        # self.scene.max_episode_length_s = 30.0  # Longer episodes
        self.scene.num_envs = 8192            # Fewer environments for faster iteration
        # self.reward.momentum_efficiency.weight = 0.7  # Increase weight of custom reward


@configclass
class CustomG1FlatAgentCfg(BaseAgentCfg):
    """Custom G1 flat agent configuration with all parameters exposed."""
    
    experiment_name: str = "custom_g1_flat"
    wandb_project: str = "custom_g1_flat"
    
    # Full control over PPO parameters
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 10000
    empirical_normalization = False
    
    # Policy network configuration
    policy_class_name = "ActorCritic"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Customize policy parameters
        self.policy.class_name = "ActorCritic"
        self.policy.init_noise_std = 1.0
        self.policy.noise_std_type = "scalar"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        
        # Customize algorithm parameters
        self.algorithm.class_name = "PPO"
        self.algorithm.value_loss_coef = 1.0
        self.algorithm.use_clipped_value_loss = True
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 8
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.desired_kl = 0.01
        self.algorithm.max_grad_norm = 1.0
        self.algorithm.normalize_advantage_per_mini_batch = False
        
        # Training configuration
        self.save_interval = 100
        self.clip_actions = None
        self.logger = "wandb"
        self.resume = False
        self.load_run = ".*"
        self.load_checkpoint = "model_.*.pt"


@configclass
class CustomG1RoughEnvCfg(CustomG1FlatEnvCfg):
    """Custom G1 rough environment configuration with all parameters exposed."""
    
    def __post_init__(self):
        super().__post_init__()
        
        # Copy settings from G1RoughEnvCfg
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.reward.feet_air_time.weight = 0.25
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 1.5
        self.reward.lin_vel_z_l2.weight = -0.25
        
        # You can customize any parameters specific to rough terrain
        # For example:
        # self.reward.momentum_efficiency.weight = 0.8  # Increase weight for rough terrain


@configclass
class CustomG1RoughAgentCfg(CustomG1FlatAgentCfg):
    """Custom G1 rough agent configuration with all parameters exposed."""
    
    experiment_name: str = "custom_g1_rough"
    wandb_project: str = "custom_g1_rough"
    
    def __post_init__(self):
        super().__post_init__()
        
        # Copy settings from G1RoughAgentCfg
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
        
        # You can customize any parameters specific to the rough terrain agent
        # For example:
        # self.algorithm.learning_rate = 5.0e-4  # Lower learning rate for stability