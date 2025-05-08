# G1-RL

Reinforcement learning training pipeline for the G1 humanoid robot, built using the [MuJoCo Playground](https://github.com/google-deepmind/mujoco-playground) and Brax PPO framework.

This repo contains a training + evaluation loop designed for joystick-commanded locomotion in simulation, using custom reward shaping and full rendering/evaluation tools. It'll later be extended to include a full train, sim-to-sim then sim-to-real deployment pipeline.

## Getting Started

You must have a working Conda environment **or** Docker container with MuJoCo Playground and its dependencies installed. This setup is GPU-accelerated and assumes proper JAX and EGL rendering support.

### Example: Conda Setup

```bash
conda create -n g1 python=3.10
conda activate g1
pip install -r requirements.txt  # Look in the MuJoCo playground repo
```

### Docker (Recommended)

For maximum reproducibility, consider using a containerized setup with:

- NVIDIA GPU support (`nvidia-docker`)
- OpenGL / EGL support for offscreen rendering
- Python 3.10+ and MuJoCo Playground installed

## Training

To train and evaluate a policy:

```bash
python run_g1_train_eval.py --gif
```

This script trains a PPO agent using the joystick locomotion environment, then renders and optionally saves a `.gif` of the final rollout.

### Output:
- Checkpoints are saved in `checkpoints/`
- Training progress plots and metrics are saved per checkpoint
- A `rollout.gif` is saved if `--gif` is passed

## Hardware Used

Trained and tested on:

- GPU: NVIDIA RTX 4090
- CPU: Intel Core i9-12900K
- RAM: 64 GB
- OS: Ubuntu 22.04 LTS

You’ll need a similar setup (or scale accordingly) to achieve high-performance simulation and training.

## TODO

- [ ] Integrate new rewards and tune them for locomotion
- [ ] Deployment script  
  - [ ] Hardware safety stack to ensure joint limits aren't reached
- [ ] Test on hardware
- [ ] Docker image for self-contained development

## Initial Results (rough)

<img src="docs/rollout_initial_testing.gif" loop=infinite>
