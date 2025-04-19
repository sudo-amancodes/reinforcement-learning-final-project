# rl-stable-retro-ppo

A lightweight implementation of Proximal Policy Optimization (PPO) for Atari‑style retro games (via `gym-retro`) and classic OpenAI Gym environments, using PyTorch and Stable Baselines3 utilities.

---

## Features

- **Custom PPOAgent**  
  - From-scratch PyTorch implementation of PPO (actor‑critic, clipped objective).  
  - Vectorized rollouts, discounted returns & advantage normalization.

- **Retro Game Support**  
  - `ppo-with-stable-retro.py` wraps `gym-retro` games with DeepMind‑style preprocessing (`WarpFrame`, `ClipRewardEnv`, frame‑stacking, stochastic frame‑skip).

- **Gym Task Example**  
  - `ppo.py` demonstrates training/testing on standard Gym environments (CartPole, MountainCar, etc.).  
  - Utilities to save/load model weights and render test episodes.

---

## Requirements

- Python 3.8+  
- PyTorch  
- gymnasium  
- gym-retro  
- stable-baselines3  
- numpy  

Install via:

```bash
pip install -r requirements.txt
