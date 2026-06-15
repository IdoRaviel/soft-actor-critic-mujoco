# Soft Actor-Critic (SAC) — From Scratch in PyTorch

A clean, from-scratch implementation of **Soft Actor-Critic** (Haarnoja et al., [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)) applied to continuous-control MuJoCo environments.

> **Course project** — Reinforcement Learning, Bar-Ilan University (2026).
> Implemented by Ido Raviel & Dan Butensky.

---

<p align="center">
  <img src="videos/ant.gif" alt="Trained Ant-v4 agent" width="500">
</p>

---

## Overview

This project implements SAC end-to-end in PyTorch with no external RL libraries. Every component — the actor, critics, replay buffer, and training loop — is built directly from the paper's equations, using only PyTorch and Gymnasium.

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
| `Ant-v4`     | 27 | 8 | [-1, 1] | 1000 |

---

## Setup

Requires **Python 3.9+** (developed on 3.11).

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

All hyperparameters live in `sac/config.py` and are overridden per-environment in `configs/`.

```bash
# Reacher-v4
python scripts/train.py --config configs/reacher.yaml --seed 0

# Pusher-v4
python scripts/train.py --config configs/pusher.yaml --seed 0

# Ant-v4
python scripts/train.py --config configs/ant.yaml --seed 0
```

Each run saves to `log/<env_id>/seed<seed>_<timestamp>/` containing:
- `config.yaml` — the fully resolved hyperparameters used
- `eval.csv` — evaluation scores every 10k steps (the learning curve)
- `best.pt` — checkpoint with the highest evaluation return
- `final.pt` — checkpoint at the end of training

---

## Inference

A trained checkpoint for each environment is provided under `pretrained_models/`, so
the agents can be run without training first.

**Watch a trained agent live** (opens the MuJoCo viewer; runs until `Ctrl+C`):
```bash
python scripts/simulate.py --checkpoint pretrained_models/Ant-v4/best.pt
```

**Score one or more checkpoints** (prints per-run mean ± std and their average), and
optionally record an mp4:
```bash
python scripts/evaluate.py --runs pretrained_models/Reacher-v4 --episodes 100
python scripts/evaluate.py --runs log/Ant-v4/seed0_* log/Ant-v4/seed1_* log/Ant-v4/seed2_* --episodes 100
python scripts/evaluate.py --runs pretrained_models/Pusher-v4 --episodes 100 --video videos/pusher.mp4
```

---

## Evaluation Protocol

- Evaluation runs every **10,000 training steps** using the **deterministic policy** (mean action, no sampling)
- During training, each evaluation averages **10 episodes**; final reported results are scored over **100 episodes**
- Final results are reported as **3 independent seeds** per environment
- Evaluation episodes use a different seed than training in both the training loop and `evaluate.py` — the policy is deterministic, but initial conditions (e.g. body pose, object/target placement) are randomized per episode, so seed diversity ensures an unbiased measure of performance

---

## Results

Final performance, scored over **100 evaluation episodes** with the deterministic policy
across **3 independent seeds** (mean return per environment):

| Environment | Seed runs | Average |
|-------------|-----------|---------|
| `Reacher-v4` | -4.11, -4.10, -4.06 | **-4.09** |
| `Pusher-v4`  | -21.75, -20.27, -20.75 | **-20.92** |
| `Ant-v4`     | 6314.91, 7113.52, 6757.49 | **6728.64** |

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
  simulate.py           #   watch a checkpoint live in the MuJoCo viewer (or save a GIF)
  plot.py               #   eval.csv → learning-curve figures
configs/
  reacher.yaml          #   per-environment hyperparameter overrides
  pusher.yaml
  ant.yaml
pretrained_models/      # one trained checkpoint (+ its config) per environment
report/                 # learning-curve figures and final evaluation results
videos/                 # recorded agent rollouts
```

---

## Reference

Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018).
**Soft Actor-Critic Algorithms and Applications.** [arXiv:1812.05905](https://arxiv.org/abs/1812.05905).
