"""
Additional custom reward functions for Unitree G1 robot.
"""

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def momentum_efficiency(env: 'BaseEnv', asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Custom reward that encourages momentum-based efficiency.
    
    This reward encourages maintaining momentum while using less energy,
    which results in more natural and efficient gaits.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration specifying the robot asset
        
    Returns:
        torch.Tensor: Reward values (one per environment)
    """
    # Access the articulation asset (robot)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get linear velocity and angular velocity
    lin_vel = asset.data.root_lin_vel_b  # Linear velocity in body frame
    ang_vel = asset.data.root_ang_vel_b  # Angular velocity in body frame
    
    # Calculate energy used (approximation through joint torques and velocities)
    energy_used = torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    
    # Get forward momentum (x-direction in body frame)
    momentum = torch.abs(lin_vel[:, 0])
    
    # Calculate efficiency as momentum per unit energy
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    efficiency = momentum / (energy_used + epsilon)
    
    # Return the efficiency value (higher is better)
    return efficiency


def gait_regularity(env: 'BaseEnv', sensor_cfg: SceneEntityCfg, cycle_time: float = 0.8) -> torch.Tensor:
    """
    Reward function that encourages regular, rhythmic gait patterns.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor
        cycle_time: Expected duration of a full gait cycle in seconds
        
    Returns:
        torch.Tensor: Reward values
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact state transitions (feet touching/leaving ground)
    # We're interested in the timing between these events
    contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    
    # Calculate the time since the last contact change
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    # Calculate variance in contact timing - lower variance means more regular gait
    contact_time_variance = torch.var(contact_time, dim=1)
    
    # Reward is higher when variance is lower (more regular gait)
    # Normalize to [0,1] range
    max_variance = cycle_time * cycle_time / 4  # Maximum expected variance
    normalized_variance = torch.clamp(contact_time_variance / max_variance, 0.0, 1.0)
    
    # Return 1 - normalized_variance so higher reward means more regular gait
    return 1.0 - normalized_variance


def foot_clearance(env: 'BaseEnv', asset_cfg: SceneEntityCfg, min_height: float = 0.05, max_height: float = 0.15) -> torch.Tensor:
    """
    Reward function that encourages appropriate foot clearance during swing phase.
    
    Too low clearance risks stumbling, while too high clearance wastes energy.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for robot asset
        min_height: Minimum desired foot clearance (meters)
        max_height: Maximum desired foot clearance (meters)
        
    Returns:
        torch.Tensor: Reward values
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get foot heights relative to ground
    foot_positions = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # z-coordinate
    ground_height = torch.zeros_like(foot_positions)  # Assuming flat ground at z=0
    
    # Calculate foot clearance
    clearance = foot_positions - ground_height
    
    # Calculate reward based on clearance
    # Reward is 1.0 when clearance is between min_height and max_height
    # Linearly decreases as it goes outside this range
    low_penalty = torch.clamp((clearance - min_height) / min_height, 0.0, 1.0)
    high_penalty = torch.clamp((max_height - clearance) / max_height, 0.0, 1.0)
    
    # Combine penalties
    reward = low_penalty * high_penalty
    
    return reward

def stance_straight_knee(env: 'BaseEnv', sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward function that encourages straight knees during stance phase.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (for feet contact detection)
        asset_cfg: Configuration for robot asset (for knee joint angles)
        
    Returns:
        torch.Tensor: Penalty values for bent knees during stance
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact states for feet
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    stance_mask = torch.norm(contact_forces, dim=-1) > 30.0  # Consider in stance if force > 1N
    
    # Get knee joint angles
    knee_angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Penalize bent knees only during stance
    penalty = torch.where(
        stance_mask,
        torch.square(knee_angles - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        torch.zeros_like(knee_angles)
    )
    
    # Sum across knees
    penalty = torch.sum(penalty, dim=1)
    
    # For zero command, we still want straight knees while standing
    # So we don't zero out this reward
    
    return penalty

def swing_knee_flexion(env: 'BaseEnv', sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, target_angle: float = 0.5) -> torch.Tensor:
    """
    Reward function that encourages proper knee flexion during swing phase.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (for feet contact detection)
        asset_cfg: Configuration for robot asset (for knee joint angles)
        target_angle: Target knee flexion angle in radians
        
    Returns:
        torch.Tensor: Reward values for proper knee flexion during swing
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact states for feet
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    swing_mask = torch.norm(contact_forces, dim=-1) < 10.0  # Consider in swing if force < 1N
    
    # Get knee joint angles
    knee_angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # Calculate reward based on proximity to target angle
    angle_error = knee_angles - target_angle
    flexion_reward = torch.exp(-torch.square(angle_error))
    
    # Only reward during swing phase
    reward = torch.where(swing_mask, flexion_reward, torch.zeros_like(flexion_reward))
    
    # Sum across knees
    reward = torch.sum(reward, dim=1)
    
    # No reward for zero command - we don't want knee flexion when standing still
    reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    
    return reward

def heel_strike_with_vel(
    env: 'BaseEnv',
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    velocity_threshold: float = -0.2,
    pitch_threshold: float = -0.075
) -> torch.Tensor:
    """
    Reward function that encourages proper heel strike with appropriate foot orientation and velocity.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (for foot contact detection)
        asset_cfg: Configuration for robot asset (for foot velocities)
        ankle_cfg: Configuration for ankle pitch joints
        velocity_threshold: Threshold for downward velocity (negative value)
        pitch_threshold: Ankle pitch angle for heel-first contact (negative = heel down)
        
    Returns:
        torch.Tensor: Reward values for proper heel strikes
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact states
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.norm(contact_forces, dim=-1) > 1.0
    
    # Get foot velocities
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    vertical_vel = foot_velocities[..., 2]  # z-axis velocity
    
    # Get ankle pitch angles (negative = heel down, positive = toes down)
    ankle_angles = asset.data.joint_pos[:, ankle_cfg.joint_ids]
    
    # Check for proper heel strike:
    # 1. Foot is pitched for heel contact (ankle angle < threshold, heel down)
    # 2. Has appropriate downward velocity (vel < threshold)
    # 3. Making initial contact
    proper_orientation = ankle_angles < pitch_threshold  # Heel down position
    downward_strike = vertical_vel < velocity_threshold  # Moving downward
    
    # Reward when all conditions are met
    reward = torch.where(
        in_contact & proper_orientation & downward_strike,
        torch.ones_like(vertical_vel),
        torch.zeros_like(vertical_vel)
    )
    
    # Sum across feet
    reward = torch.sum(reward, dim=1)
    
    # No reward for zero command
    reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    
    return reward


def toe_off_with_vel(
    env: 'BaseEnv',
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    ankle_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.2,
    pitch_threshold: float = 0.075
) -> torch.Tensor:
    """
    Reward function that encourages proper toe-off with appropriate foot orientation and velocity.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (for foot contact detection)
        asset_cfg: Configuration for robot asset (for foot velocities)
        ankle_cfg: Configuration for ankle pitch joints
        velocity_threshold: Threshold for upward velocity (positive value)
        pitch_threshold: Ankle pitch angle for toe-off (positive = toes down)
        
    Returns:
        torch.Tensor: Reward values for proper toe-offs
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get contact states
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.norm(contact_forces, dim=-1) > 1.0
    
    # Get foot velocities
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    vertical_vel = foot_velocities[..., 2]  # z-axis velocity
    
    # Get ankle pitch angles (negative = heel down, positive = toes down)
    ankle_angles = asset.data.joint_pos[:, ankle_cfg.joint_ids]
    
    # Check for proper toe-off:
    # 1. Foot is pitched for toe push (ankle angle > threshold, toes down)
    # 2. Has appropriate upward velocity (vel > threshold)
    # 3. Breaking contact
    proper_orientation = ankle_angles > pitch_threshold  # Toes down position
    upward_push = vertical_vel > velocity_threshold     # Moving upward
    
    # Reward when all conditions are met
    reward = torch.where(
        ~in_contact & proper_orientation & upward_push,
        torch.ones_like(vertical_vel),
        torch.zeros_like(vertical_vel)
    )
    
    # Sum across feet
    reward = torch.sum(reward, dim=1)
    
    # No reward for zero command
    reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    
    return reward

def smooth_foot_landing(
    env: 'BaseEnv',
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    velocity_threshold: float = 0.5
) -> torch.Tensor:
    """
    Reward function that encourages smooth foot landing by penalizing high vertical velocities
    near ground contact.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for contact sensor (for foot contact detection)
        asset_cfg: Configuration for robot asset (for foot velocities)
        velocity_threshold: Maximum allowed vertical velocity magnitude (default: 0.5 m/s)
        
    Returns:
        torch.Tensor: Penalty values for high-velocity landings
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get foot velocities
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    vertical_vel = foot_velocities[..., 2]  # z-axis velocity
    
    # Get contact states
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    near_contact = torch.norm(contact_forces, dim=-1) > 0.1  # Very light contact threshold
    
    # Calculate penalty for high velocities when near contact
    # Square the excess velocity for quadratic penalty
    excess_vel = torch.clamp(torch.abs(vertical_vel) - velocity_threshold, min=0.0)
    penalty = torch.square(excess_vel)
    
    # Only apply penalty when foot is near contact
    penalty = torch.where(near_contact, penalty, torch.zeros_like(penalty))
    
    # Sum across feet
    penalty = torch.sum(penalty, dim=1)
    
    # Only apply penalty when command is non-zero
    penalty *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > 0.1
    
    return penalty
