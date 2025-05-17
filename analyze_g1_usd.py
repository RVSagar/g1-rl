"""
Script to analyze the G1 USD model and print joint ordering and model data.
This helps verify joint order matches between IsaacSim and Mujoco.
"""

# Launch Isaac Sim Simulator first.
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Analyze G1 USD model joint ordering.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# Rest everything follows.

# Import IsaacLab modules after app launch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg


from legged_lab.assets.unitree.unitree import G1_CFG

# Initialize simulation
sim_cfg = SimulationCfg(device=args_cli.device)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])

# Create scene config with G1 robot
scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
scene_cfg.robot = G1_CFG.replace(prim_path="/World/Robot")

# Create scene and add robot
scene = InteractiveScene(scene_cfg)
sim.reset()

# Print joint names and order
print("\nJoint Names in Order:")
robot = scene["robot"]
for i, name in enumerate(robot.data.joint_names):
    print(f"{i}: {name}")

print("\nKeeping viewer open. Press Ctrl+C to exit.")

# Keep the viewer open
while simulation_app.is_running():
    sim.step()
