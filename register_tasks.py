"""
Register and run training with custom G1 configurations.
"""

import os
import sys
import torch

# Make sure LeggedLab is installed
try:
    import legged_lab
except ImportError:
    raise ImportError(
        "LeggedLab not found. Please install it first."
    )

# Import our custom configurations
from g1_locomotion_config import (
    CustomG1FlatEnvCfg, 
    CustomG1FlatAgentCfg,
    CustomG1RoughEnvCfg,
    CustomG1RoughAgentCfg
)

# Import LeggedLab utilities
# from legged_lab.utils.helpers import get_args, update_cfg_from_args, class_to_dict
from legged_lab.utils.task_registry import task_registry
from legged_lab.envs.base.base_env import BaseEnv

# Register our custom tasks
def register_custom_tasks():
    """Register custom tasks with LeggedLab's task registry."""
    
    # Register custom G1 flat task
    task_registry.register(
        name="custom_g1_flat",
        task_class=BaseEnv,
        env_cfg=CustomG1FlatEnvCfg(),
        train_cfg=CustomG1FlatAgentCfg()
    )
    
    # Register custom G1 rough task
    task_registry.register(
        name="custom_g1_rough",
        task_class=BaseEnv,
        env_cfg=CustomG1RoughEnvCfg(),
        train_cfg=CustomG1RoughAgentCfg()
    )
    
    print("Custom tasks registered successfully!")
    
    # List available tasks
    print("\nAvailable tasks:")
    for task_name in task_registry.task_classes.keys():
        print(f"  - {task_name}")


# Register the tasks
register_custom_tasks()

# If this script is run directly, print usage instructions
if __name__ == "__main__":
    print("\nTo start training with your custom tasks, run:")
    print("  python legged_lab/scripts/train.py --task=custom_g1_flat")
    print("  python legged_lab/scripts/train.py --task=custom_g1_rough")
    print("\nTo visualize a trained policy:")
    print("  python legged_lab/scripts/play.py --task=custom_g1_flat")
    print("  python legged_lab/scripts/play.py --task=custom_g1_rough")