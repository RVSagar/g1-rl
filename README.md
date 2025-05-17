# G1-RL: Sim-to-Real Pipeline for G1 Humanoid

A comprehensive training and deployment pipeline for the Unitree G1 humanoid robot, focusing on robust sim-to-real transfer through multi-simulator validation.

## Sim-to-Real Pipeline

| **IsaacLab Training** | **MuJoCo Testing** | **Real Deployment** |
|:---------------------:|:------------------:|:-------------------:|
| ![](images/isaaclab_flat_g1.gif) | ![](images/mujoco_sim_sim_test_flat_g1.gif) | *Coming soon...* |



## Overview

This repository implements a three-stage pipeline for developing robust locomotion policies for the G1 humanoid:
1. Training in IsaacLab with custom rewards and domain randomization
2. Validation in MuJoCo to ensure sim-to-sim transfer
3. Deployment to real hardware with safety constraints

The deployment code was inspired by [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) and [LeggedLabDeploy](https://github.com/Hellod035/LeggedLabDeploy). Further commits to come once cleaned up.

## Key Features

- Custom reward shaping for natural locomotion
  - Heel-strike and toe-off rewards
  - Energy efficiency and smoothness terms
  - Knee and ankle coordination rewards
- Multi-simulator validation pipeline
- Hardware-ready deployment tools
- Comprehensive configuration system

## Repository Structure

```
g1-rl/
├── train_custom.py          # Main training script for IsaacLab
├── play_custom.py           # Policy visualization in IsaacLab
├── deploy_mujoco.py         # MuJoCo deployment and testing
├── view_g1_mujoco.py        # MuJoCo visualization tools
├── g1_locomotion_config.py  # Training configuration
├── custom_rewards.py        # Custom reward functions
├── g1_config_loader.py      # Hardware config management
├── analyze_g1_usd.py        # Joint order analysis tools
└── register_tasks.py        # Environment registration
```

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- IsaacLab and LeggedLab installations
- MuJoCo 3.0+
- Python 3.10+

### Installation

TODO

## Training Pipeline

1. **IsaacLab Training**
```bash
python train_custom.py --max_iterations=10000
```

2. **IsaacLab Validation**
```bash
python play_custom.py  # plays latest run
```

3. **MuJoCo Validation**
```bash
python deploy_mujoco.py --policy_path path/to/policy
```

## Configuration

The training pipeline is highly configurable through `g1_locomotion_config.py`:
- Custom reward terms and weights
- PD gains and joint limits
- Training hyperparameters
- Domain randomization settings

## Hardware Deployment

The deployment pipeline includes:
- Joint order mapping between simulators
- PD gain matching
- Hardware safety constraints
- Real-time observation scaling

## TODO

- [ ] Dockerfile for better reproducibility
- [ ] Submodules for IsaacLab and LeggedLab
- [ ] Code refactoring for better organization and workflow
- [ ] Comprehensive testing suite
- [ ] Real robot deployment safety checks
- [ ] Documentation for reward tuning
- [ ] Training metrics visualization
- [ ] Improve heel-strike and toe-off rewards for more natural gait
