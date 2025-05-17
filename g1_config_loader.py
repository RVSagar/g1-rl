"""
Configuration loader for G1 robot deployment.
Contains default positions and gains from LeggedLab's G1_CFG without requiring IsaacLab dependencies.
"""

import numpy as np

def get_default_joint_positions():
    """Get default joint positions in Mujoco order."""
    # Initialize array with zeros
    default_positions = np.zeros(29)
    
    # Fill in the known defaults from G1_CFG.init_state.joint_pos
    # Values from LeggedLab/legged_lab/assets/unitree/unitree.py
    hip_pitch = -0.20
    knee = 0.42
    ankle_pitch = -0.23
    elbow = 0.87
    left_shoulder_roll = 0.18
    right_shoulder_roll = -0.18
    shoulder_pitch = 0.35
    
    # Map to Mujoco order
    default_positions[0] = hip_pitch      # left hip pitch
    default_positions[6] = hip_pitch      # right hip pitch
    default_positions[3] = knee           # left knee
    default_positions[9] = knee           # right knee
    default_positions[4] = ankle_pitch    # left ankle pitch
    default_positions[10] = ankle_pitch   # right ankle pitch
    default_positions[18] = elbow         # left elbow
    default_positions[25] = elbow         # right elbow
    default_positions[16] = left_shoulder_roll   # left shoulder roll
    default_positions[23] = right_shoulder_roll  # right shoulder roll
    default_positions[15] = shoulder_pitch       # left shoulder pitch
    default_positions[22] = shoulder_pitch       # right shoulder pitch
    
    return default_positions

def get_gains():
    """Get PD gains in Mujoco order."""
    # Initialize gain arrays
    p_gains = np.zeros(29)
    d_gains = np.zeros(29)
    
    # Helper to set gains for matching joint patterns
    def set_gains_for_pattern(p_gain, d_gain, joint_indices):
        for idx in joint_indices:
            p_gains[idx] = p_gain
            d_gains[idx] = d_gain
    
    # Values from LeggedLab/legged_lab/assets/unitree/unitree.py
    
    # Legs
    set_gains_for_pattern(150.0, 5.0, [2, 8])   # hip yaw
    set_gains_for_pattern(150.0, 5.0, [1, 7])   # hip roll
    set_gains_for_pattern(200.0, 5.0, [0, 6])   # hip pitch
    set_gains_for_pattern(200.0, 5.0, [3, 9])   # knee
    
    # Waist
    set_gains_for_pattern(200.0, 5.0, [12, 13, 14])  # waist yaw, roll, pitch
    
    # Feet
    set_gains_for_pattern(20.0, 2.0, [4, 10])   # ankle pitch
    set_gains_for_pattern(20.0, 2.0, [5, 11])   # ankle roll
    
    # Shoulders
    set_gains_for_pattern(100.0, 2.0, [15, 22])  # shoulder pitch
    set_gains_for_pattern(100.0, 2.0, [16, 23])  # shoulder roll
    
    # Arms
    set_gains_for_pattern(50.0, 2.0, [17, 24])   # shoulder yaw
    set_gains_for_pattern(50.0, 2.0, [18, 25])   # elbow
    
    # Wrists
    set_gains_for_pattern(40.0, 2.0, [19, 26])   # wrist roll
    set_gains_for_pattern(40.0, 2.0, [20, 27])   # wrist pitch
    set_gains_for_pattern(40.0, 2.0, [21, 28])   # wrist yaw
    
    return p_gains, d_gains 