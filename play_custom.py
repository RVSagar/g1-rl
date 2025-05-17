"""
Script to play/visualize a trained policy with custom G1 configurations.
This is a convenient wrapper around LeggedLab's play.py
that automatically registers the custom tasks.
"""

import os
import sys
import argparse

#the order of imports is important, need to import play_module before legged_lab_play before register_custom_tasks
import legged_lab.scripts.play as play_module
from legged_lab.scripts.play import play as legged_lab_play

# Register custom tasks first
from register_tasks import register_custom_tasks
register_custom_tasks()

def main():
    """Parse args and run visualization with custom tasks."""
    
    # Add our own argument parsing
    parser = argparse.ArgumentParser(description="Play/visualize with custom G1 configurations")
    parser.add_argument("--task", type=str, default="custom_g1_flat", 
                      help="Task to visualize (custom_g1_flat or custom_g1_rough)")
    parser.add_argument("--rough", action="store_true", 
                      help="Use rough terrain configuration (shorthand for --task=custom_g1_rough)")
    
    # Add run selection arguments
    parser.add_argument("--run", type=str, default=".*",
                      help="Regular expression to match the run directory name (default: '.*' matches any)")
    parser.add_argument("--checkpoint", type=str, default="model_.*.pt",
                      help="Regular expression to match the checkpoint file (default: 'model_.*.pt' matches any model file)")
    
    # Add all the arguments that play.py needs
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--device", type=str, default=None, help="Device to run on")
    
    # Parse our arguments
    args = parser.parse_args()
    
    # Set task based on our args
    if args.rough:
        args.task = "custom_g1_rough"
    
    # Print visualization configuration
    print(f"\nVisualizing policy for task: {args.task}")
    print(f"Device: {args.device if args.device else 'default'}")
    print(f"Number of environments: {args.num_envs if args.num_envs else 'default'}")
    print(f"Headless mode: {args.headless}")
    print(f"Looking for run matching: {args.run}")
    print(f"Looking for checkpoint matching: {args.checkpoint}")
    
    # Import play.py - it will parse args but we'll override them directly
    import legged_lab.scripts.play as play_module
    from legged_lab.scripts.play import play as legged_lab_play
    
    # Modify the args_cli variable directly in the play module
    play_module.args_cli.task = args.task
    if args.num_envs is not None:
        play_module.args_cli.num_envs = args.num_envs
    if args.seed is not None:
        play_module.args_cli.seed = args.seed
    play_module.args_cli.headless = args.headless
    if args.device is not None:
        play_module.args_cli.device = args.device
    
    # Set the run and checkpoint patterns
    play_module.args_cli.load_run = args.run
    play_module.args_cli.load_checkpoint = args.checkpoint
    
    # Call the play function from LeggedLab
    legged_lab_play()

if __name__ == "__main__":
    main()