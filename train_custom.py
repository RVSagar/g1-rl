"""
Script to train with custom G1 configurations.
This is a convenient wrapper around LeggedLab's train.py
that automatically registers the custom tasks.
"""
import os
import sys
import argparse

#IsaacLab import order problems for registering tasks below (missing omni.kit error): https://github.com/isaac-sim/IsaacLab/issues/927
from legged_lab.scripts.train import train as legged_lab_train

# Register custom tasks first
from register_tasks import register_custom_tasks
register_custom_tasks()

def main():
    """Parse args and run training with custom tasks."""
    
    # Add our own argument parsing
    parser = argparse.ArgumentParser(description="Train with custom G1 configurations")
    parser.add_argument("--task", type=str, default="custom_g1_flat", 
                      help="Task to train (custom_g1_flat or custom_g1_rough)")
    parser.add_argument("--rough", action="store_true", 
                      help="Use rough terrain configuration (shorthand for --task=custom_g1_rough)")
    
    # Add all the arguments that train.py needs as well
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of training iterations")
    
    # Parse our arguments
    args = parser.parse_args()
    
    # Set task based on our args
    if args.rough:
        args.task = "custom_g1_rough"
    
    # Print training configuration
    print(f"\nStarting training with task: {args.task}")
    print(f"Device: {args.device if args.device else 'default'}")
    print(f"Number of environments: {args.num_envs if args.num_envs else 'default'}")
    print(f"Max iterations: {args.max_iterations if args.max_iterations else 'default'}")
    print(f"Headless mode: {args.headless}")
    
    # Import train.py - it will parse args but we'll override them directly
    import legged_lab.scripts.train as train_module
    from legged_lab.scripts.train import train as legged_lab_train
    
    # Modify the args_cli variable directly in the train module
    train_module.args_cli.task = args.task
    if args.num_envs is not None:
        train_module.args_cli.num_envs = args.num_envs
    if args.seed is not None:
        train_module.args_cli.seed = args.seed
    train_module.args_cli.headless = args.headless
    if args.device is not None:
        train_module.args_cli.device = args.device
    if args.max_iterations is not None:
        train_module.args_cli.max_iterations = args.max_iterations
    
    # Call the train function from LeggedLab
    legged_lab_train()

if __name__ == "__main__":
    main()