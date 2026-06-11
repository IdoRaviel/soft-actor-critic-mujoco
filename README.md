# Soft Actor-Critic (SAC) — From Scratch in PyTorch

A clean, from-scratch implementation of **Soft Actor-Critic** (Haarnoja et al., [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)) applied to continuous-control MuJoCo environments.

> **Course project** — Reinforcement Learning, Bar-Ilan University (2026).
> Implemented by Ido Raviel & Dan Butensky.

---

## Overview

This project implements SAC end-to-end in PyTorch with no external RL libraries. Every component — the actor, critics, replay buffer, and training loop — is hand-written from the paper.

**Key algorithmic features:**
- Twin Q-critics with Polyak-averaged target networks (clipped double-Q)
- Tanh-squashed Gaussian policy with reparameterized sampling and exact log-prob correction
- Automatic entropy temperature tuning (learned `alpha`)
- Separate `terminated` / `truncated` handling for correct value bootstrapping

**Environments (MuJoCo v4 via Gymnasium):**

| Environment | Observation | Actions | Action range | Episode steps |
|-------------|-------------|---------|--------------|---------------|
| `Reacher-v4` | 11 | 2 | [-1, 1] | 50 |
| `Pusher-v4`  | 23 | 7 | [-2, 2] | 100 |

---

## Project Structure

```
sac/                    # algorithm library — no RL dependencies
  config.py             #   Config dataclass + YAML loader
  networks.py           #   GaussianPolicy (actor) + QNetwork (critic)
  replay_buffer.py      #   FIFO replay buffer (numpy-backed)
  agent.py              #   SAC update loop: critics → actor → temperature → Polyak
  utils.py              #   seeding, CSV logging, run directories
scripts/
  train.py              #   training loop: eval every 10k steps, checkpoints, CSV logs
  evaluate.py           #   score a checkpoint + optional mp4 recording
  simulate.py           #   watch a checkpoint live in the MuJoCo viewer
  plot.py               #   eval.csv → learning-curve figures
configs/
  reacher.yaml          #   per-environment hyperparameter overrides
  pusher.yaml
```

---

## Setup

Requires **Python 3.11**.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the sac package in editable mode
pip install -e .
```

> `requirements.txt` defaults to the **CUDA 12.1** PyTorch build (`torch==2.5.1+cu121`).
> For CPU-only machines, replace the torch line with `torch==2.5.1`.

---

## Training

All hyperparameters live in `sac/config.py` and are overridden per-environment in `configs/`. The CLI only exposes `--config` and `--seed` — everything else is in the YAML.

```bash
# Reacher-v4
python scripts/train.py --config configs/reacher.yaml --seed 0

# Pusher-v4
python scripts/train.py --config configs/pusher.yaml --seed 0
```

Each run saves to `log/<env_id>/seed<seed>_<timestamp>/` containing:
- `config.yaml` — the fully resolved hyperparameters used
- `eval.csv` — evaluation scores every 10k steps (the learning curve)
- `best.pt` — checkpoint with the highest evaluation return
- `final.pt` — checkpoint at the end of training

---

## Inference

> **Pretrained models** — coming soon. A `pretrained/` folder with one checkpoint per environment will be added after training is complete.

**Watch a trained agent live:**
```bash
python scripts/simulate.py --checkpoint log/Reacher-v4/seed0_<timestamp>/best.pt --episodes 20
```

**Run headless evaluation and optionally record a video:**
```bash
python scripts/evaluate.py --run log/Reacher-v4/seed0_<timestamp> --episodes 10
python scripts/evaluate.py --run log/Reacher-v4/seed0_<timestamp> --episodes 10 --video videos/reacher.mp4
```

---

## Evaluation Protocol

- Evaluation runs every **10,000 training steps** using the **deterministic policy** (mean action, no sampling)
- Each evaluation averages **10 episodes**
- Final results are reported as **3 independent seeds** per environment
- Evaluation episodes use a different seed than training in both the training loop and `evaluate.py` — the policy is deterministic, but initial conditions (arm position, target position) are randomized per episode, so seed diversity ensures an unbiased measure of performance

---

## Hyperparameters

| Parameter | Reacher-v4 | Pusher-v4 |
|-----------|-----------|-----------|
| Total steps | 150,000 | 1,000,000 |
| Warmup steps | 5,000 | 10,000 |
| Hidden layers | 2 × 256 | 2 × 256 |
| Learning rate | 3 × 10⁻⁴ | 3 × 10⁻⁴ |
| Discount γ | 0.99 | 0.99 |
| Polyak τ | 0.005 | 0.005 |
| Batch size | 256 | 256 |
| Replay buffer | 1,000,000 | 1,000,000 |
| Target entropy | -2 | -7 |

---

## Reference

Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018).
**Soft Actor-Critic Algorithms and Applications.** [arXiv:1812.05905](https://arxiv.org/abs/1812.05905).
