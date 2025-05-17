"""
Script to deploy and test G1 policies in Mujoco before real hardware deployment.
This allows testing sim2sim transfer and policy behavior in a controlled environment.
"""

import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pynput import keyboard
import os
from collections import deque

from g1_config_loader import get_default_joint_positions, get_gains

# Global command state
XYYAW = {"x": 0.0, "y": 0.0, "yaw": 0.0}

# Command ranges from LeggedLabDeploy
CMD_RANGES = {
    "lin_vel_x": [-0.4, 0.7],    # Forward/backward
    "lin_vel_y": [-0.4, 0.4],    # Left/right
    "ang_vel_z": [-1.57, 1.57]   # Yaw
}

# Command scaling
CMD_SCALE = np.array([2.0, 2.0, 0.25])  # From LeggedLabDeploy

# IsaacLab to Mujoco joint order mapping
# This maps from the policy output order to Mujoco joint order
ISAACLAB_TO_MUJOCO_IDX = {
    # Legs
    0: 0,   # left_hip_pitch_joint
    1: 6,   # right_hip_pitch_joint
    3: 1,   # left_hip_roll_joint
    4: 7,   # right_hip_roll_joint
    6: 2,   # left_hip_yaw_joint
    7: 8,   # right_hip_yaw_joint
    9: 3,   # left_knee_joint
    10: 9,  # right_knee_joint
    13: 4,  # left_ankle_pitch_joint
    14: 10, # right_ankle_pitch_joint
    17: 5,  # left_ankle_roll_joint
    18: 11, # right_ankle_roll_joint
    # Torso
    2: 12,  # waist_yaw_joint
    5: 13,  # waist_roll_joint
    8: 14,  # waist_pitch_joint
    # Arms
    11: 15, # left_shoulder_pitch_joint
    12: 22, # right_shoulder_pitch_joint
    15: 16, # left_shoulder_roll_joint
    16: 23, # right_shoulder_roll_joint
    19: 17, # left_shoulder_yaw_joint
    20: 24, # right_shoulder_yaw_joint
    21: 18, # left_elbow_joint
    22: 25, # right_elbow_joint
    23: 19, # left_wrist_roll_joint
    24: 26, # right_wrist_roll_joint
    25: 20, # left_wrist_pitch_joint
    26: 27, # right_wrist_pitch_joint
    27: 21, # left_wrist_yaw_joint
    28: 28, # right_wrist_yaw_joint
}

# Control instructions
control_guide = """
🔹 Robot Keyboard Control Guide 🔹
---------------------------------
[8]  Move Forward   [2]  Move Backward
[4]  Move Left      [6]  Move Right
[7]  Turn Left      [9]  Turn Right
[5]  Stop
[ESC] Exit Program
---------------------------------
"""

def on_press(key):
    """Handle keyboard input for velocity commands."""
    try:
        if key.char == "8":  # Forward
            XYYAW["x"] += 0.05
            XYYAW["x"] = np.clip(XYYAW["x"], CMD_RANGES["lin_vel_x"][0], CMD_RANGES["lin_vel_x"][1])
            print("cmd:", XYYAW)
        elif key.char == "2":  # Backward
            XYYAW["x"] -= 0.05
            XYYAW["x"] = np.clip(XYYAW["x"], CMD_RANGES["lin_vel_x"][0], CMD_RANGES["lin_vel_x"][1])
            print("cmd:", XYYAW)
        elif key.char == "4":  # Left
            XYYAW["y"] += 0.05
            XYYAW["y"] = np.clip(XYYAW["y"], CMD_RANGES["lin_vel_y"][0], CMD_RANGES["lin_vel_y"][1])
            print("cmd:", XYYAW)
        elif key.char == "6":  # Right
            XYYAW["y"] -= 0.05
            XYYAW["y"] = np.clip(XYYAW["y"], CMD_RANGES["lin_vel_y"][0], CMD_RANGES["lin_vel_y"][1])
            print("cmd:", XYYAW)
        elif key.char == "7":  # Turn left
            XYYAW["yaw"] += 0.05
            XYYAW["yaw"] = np.clip(XYYAW["yaw"], CMD_RANGES["ang_vel_z"][0], CMD_RANGES["ang_vel_z"][1])
            print("cmd:", XYYAW)
        elif key.char == "9":  # Turn right
            XYYAW["yaw"] -= 0.05
            XYYAW["yaw"] = np.clip(XYYAW["yaw"], CMD_RANGES["ang_vel_z"][0], CMD_RANGES["ang_vel_z"][1])
            print("cmd:", XYYAW)
        elif key.char == "5":  # Stop
            XYYAW["x"] = 0.0
            XYYAW["y"] = 0.0
            XYYAW["yaw"] = 0.0
            print("cmd:", XYYAW)
    except AttributeError:
        pass

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands using PD control."""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_gravity_orientation(quaternion):
    """Get gravity vector in body frame from quaternion."""
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def reorder_mujoco_to_isaaclab(mujoco_data):
    """Reorder data from Mujoco order to IsaacLab order."""
    reordered = np.zeros_like(mujoco_data)
    # Reverse the mapping: for each isaaclab_idx -> mujoco_idx, create mujoco_idx -> isaaclab_idx
    mujoco_to_isaaclab = {mujoco_idx: isaac_idx for isaac_idx, mujoco_idx in ISAACLAB_TO_MUJOCO_IDX.items()}
    for mujoco_idx, isaac_idx in mujoco_to_isaaclab.items():
        reordered[isaac_idx] = mujoco_data[mujoco_idx]
    return reordered

def reorder_actions(actions):
    """Reorder actions from IsaacLab order to Mujoco order."""
    reordered = np.zeros_like(actions)
    for isaac_idx, mujoco_idx in ISAACLAB_TO_MUJOCO_IDX.items():
        reordered[mujoco_idx] = actions[isaac_idx]
    return reordered

class DeployNode:
    """Node for deploying and testing G1 policies in Mujoco."""
    
    def __init__(self, policy_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy_path = policy_path
        
        # Load policy
        print(f"Loading policy from: {self.policy_path}")
        self.policy = torch.jit.load(self.policy_path, map_location=self.device)
        self.policy.to(self.device)
        
        # Initialize Mujoco
        xml_path = os.path.join(os.path.dirname(__file__), "assets/robots/g1/scene_29dof.xml")
        print(f"Loading Mujoco model from: {xml_path}")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.005  # 200Hz physics, matching training environment
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        
        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        
        # Initialize parameters
        self.num_actions = 29
        self.obs_dim = 96  # Base observation dimension
        self.obs_history_len = 10  # Keep history length of 10 for trained policy
        self.control_dt = 0.02  # 50Hz control (4 physics steps per control step)
        
        # Observation scaling - Match LeggedLabDeploy exactly
        self.obs_scales = {
            'ang_vel': 1.0,          # Match LeggedLabDeploy
            'dof_pos': 1.0,           # Match LeggedLabDeploy
            'dof_vel': 1.0,          # Match LeggedLabDeploy
            'actions': 1.0
        }
        
        # Action scaling
        self.action_scale = 0.25
        
        # Load default joint positions and PD gains from LeggedLab config
        self.default_dof_pos = get_default_joint_positions()
        self.p_gains, self.d_gains = get_gains()
        
        # Initialize state
        self.prev_action = np.zeros(self.num_actions)
        self.current_obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Initialize observation history
        self.obs_history = deque(maxlen=self.obs_history_len)
        empty_obs = self.get_single_observation()
        for _ in range(self.obs_history_len):
            self.obs_history.append(empty_obs)
        
        # Set initial pose
        self.mj_data.qpos[7:] = self.default_dof_pos
        self.mj_data.qpos[:3] = [0, 0, 0.78]  # Set initial height
        mujoco.mj_forward(self.mj_model, self.mj_data)
        
        # Apply initial PD control
        tau = pd_control(
            self.default_dof_pos,
            self.mj_data.qpos[7:],
            self.p_gains,
            np.zeros(self.num_actions),
            self.mj_data.qvel[6:],
            self.d_gains
        )
        self.mj_data.ctrl[:] = tau
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()
    
    def get_single_observation(self):
        """Get current observation without history."""
        # Get command from keyboard input
        cmd = np.array([XYYAW["x"], XYYAW["y"], XYYAW["yaw"]], dtype=np.float32)
        # Scale and clip commands
        cmd = cmd * CMD_SCALE
        cmd = np.clip(cmd, [CMD_RANGES["lin_vel_x"][0], CMD_RANGES["lin_vel_y"][0], CMD_RANGES["ang_vel_z"][0]], 
                          [CMD_RANGES["lin_vel_x"][1], CMD_RANGES["lin_vel_y"][1], CMD_RANGES["ang_vel_z"][1]])
        
        # Get joint states and reorder to IsaacLab order
        dof_pos = reorder_mujoco_to_isaaclab(self.mj_data.qpos[7:] - self.default_dof_pos) * self.obs_scales['dof_pos']
        dof_vel = reorder_mujoco_to_isaaclab(self.mj_data.qvel[6:]) * self.obs_scales['dof_vel']
        
        # Get base state
        quat = self.mj_data.qpos[3:7]  # Root quaternion
        ang_vel = self.mj_data.qvel[3:6] * self.obs_scales['ang_vel']  # Root angular velocity
        
        # Calculate projected gravity (no scaling needed as it's already normalized)
        gravity_orientation = get_gravity_orientation(quat)
        
        # Previous actions are already in IsaacLab order since we store them that way
        scaled_prev_action = self.prev_action * self.obs_scales['actions']
        
        # Combine observations in LeggedLabDeploy order (96 dimensions)
        self.current_obs[:3] = ang_vel                                    # 3
        self.current_obs[3:6] = gravity_orientation                       # 3
        self.current_obs[6:9] = cmd                                       # 3
        self.current_obs[9:9+self.num_actions] = dof_pos                 # 29
        self.current_obs[9+self.num_actions:9+self.num_actions*2] = dof_vel  # 29
        self.current_obs[9+self.num_actions*2:9+self.num_actions*3] = scaled_prev_action  # 29
        
        return self.current_obs.copy()
    
    def get_observations(self):
        """Get observations with history."""
        # Get current observation
        current_obs = self.get_single_observation()
        
        # Update history
        self.obs_history.append(current_obs)
        
        # Concatenate the last 10 observations to match policy input size
        recent_history = list(self.obs_history)  # Get all 10 observations
        obs_with_history = np.concatenate(recent_history)  # This will give us 96*10 = 960 dimensions
        
        # Convert to tensor and add batch dimension
        return torch.tensor(obs_with_history, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def main_loop(self):
        """Main control loop for policy deployment."""
        print("Starting policy deployment in Mujoco...")
        print("Use the keyboard controls to move the robot")
        print("Press Ctrl+C to stop")
        
        try:
            step_count = 0
            current_action = None
            current_target_angles = None
            
            while True:
                step_start = time.time()
                
                # Get new action every 4 physics steps (50Hz control)
                if step_count % 4 == 0:
                    # Get observations and run policy
                    obs = self.get_observations()
                    with torch.no_grad():
                        actions = self.policy(obs)
                    
                    if torch.any(torch.isnan(actions)):
                        print("Emergency stop due to NaN actions")
                        break
                    
                    # Process actions - Reorder from IsaacLab to Mujoco
                    raw_actions = actions.cpu().numpy().squeeze()
                    self.prev_action = raw_actions
                    # Reorder actions from IsaacLab to Mujoco joint order
                    reordered_actions = reorder_actions(self.prev_action)
                    current_target_angles = reordered_actions * self.action_scale + self.default_dof_pos
                
                # Compute PD control
                tau = pd_control(
                    current_target_angles,
                    self.mj_data.qpos[7:],
                    self.p_gains,
                    np.zeros(self.num_actions),
                    self.mj_data.qvel[6:],
                    self.d_gains
                )
                
                # Apply control and step physics (200Hz)
                self.mj_data.ctrl[:] = tau
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.viewer.sync()
                
                step_count += 1
                
                # Maintain timing (200Hz)
                time_taken = time.time() - step_start
                if time_taken < self.mj_model.opt.timestep:
                    time.sleep(self.mj_model.opt.timestep - time_taken)
        
        except KeyboardInterrupt:
            print("\nStopping policy deployment...")
        finally:
            # Cleanup
            self.keyboard_listener.stop()
            self.viewer.close()

def main():
    """Main function to run policy deployment."""
    parser = argparse.ArgumentParser(description="Deploy G1 policy in Mujoco simulation")
    parser.add_argument('--policy_path', type=str, required=True,
                      help="Path to the exported policy file")
    args = parser.parse_args()
    
    dp_node = DeployNode(policy_path=args.policy_path)
    dp_node.main_loop()

if __name__ == "__main__":
    main() 
