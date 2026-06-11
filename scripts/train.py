"""Train a SAC agent on a MuJoCo v4 environment.

Example:
    python scripts/train.py --config configs/reacher.yaml --seed 0

Follows the assignment protocol: evaluate every ``eval_interval`` steps over
``eval_episodes`` episodes with the deterministic policy, logging the mean
return to ``log/<run_name>_<timestamp>/eval.csv`` (the learning curve for the
report). Each run writes to its own timestamped folder.
"""

from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np

from sac.agent import SACAgent
from sac.config import load_config
from sac.replay_buffer import ReplayBuffer
from sac.utils import CSVLogger, make_run_dir, set_seed


def evaluate(agent: SACAgent, env: gym.Env, episodes: int, seed_base: int) -> tuple:
    """Run ``episodes`` deterministic episodes; return (mean, std) of returns."""
    returns = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed_base + ep)
        done, total = False, 0.0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def parse_args() -> argparse.Namespace:
    # Everything is configured in the YAML/dataclass; the only per-run override
    # is the seed (the 3 required runs per env differ only by seed).
    p = argparse.ArgumentParser(description="Train SAC on a MuJoCo v4 env.")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument("--seed", type=int, default=None, help="Override config seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, seed=args.seed)
    set_seed(cfg.seed)

    # Each run gets its own timestamped folder so runs never overwrite.
    # Structure: log/<env_id>/seed<seed>_<timestamp>/
    run_id = f"seed{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = make_run_dir(os.path.join(cfg.out_dir, cfg.env_id), run_id)
    cfg.save(os.path.join(run_dir, "config.yaml"))
    logger = CSVLogger(
        os.path.join(run_dir, "eval.csv"), ["step", "return_mean", "return_std"]
    )

    # Separate envs for rollouts and evaluation.
    env = gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)
    env.action_space.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, env.action_space.low, env.action_space.high, cfg)
    buffer = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, agent.device.type)

    print(f"[{cfg.env_id} | {run_id}] device={agent.device} obs={obs_dim} act={act_dim} "
          f"steps={cfg.total_steps} eval_every={cfg.eval_interval}")

    best_return = -np.inf
    start_time = time.time()
    state, _ = env.reset(seed=cfg.seed)

    for step in range(1, cfg.total_steps + 1):
        # --- act: random warmup, then policy --------------------------------
        if step <= cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, term, trunc, _ = env.step(action)
        # Store `terminated` only: time-limit truncation must still bootstrap.
        buffer.add(state, action, reward, next_state, term)
        state = next_state
        if term or trunc:
            state, _ = env.reset()

        # --- learn ----------------------------------------------------------
        if step > cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                agent.update(buffer.sample(cfg.batch_size))

        # --- evaluate + checkpoint -----------------------------------------
        if step % cfg.eval_interval == 0:
            mean, std = evaluate(agent, eval_env, cfg.eval_episodes, cfg.seed + 10_000)
            logger.log({"step": step, "return_mean": mean, "return_std": std})
            elapsed = time.time() - start_time
            sps = step / elapsed
            print(f"step {step:>8d} | eval {mean:8.2f} +/- {std:6.2f} | "
                  f"{sps:6.0f} steps/s")
            if cfg.save_model and mean > best_return:
                best_return = mean
                agent.save(os.path.join(run_dir, "best.pt"))

    if cfg.save_model:
        agent.save(os.path.join(run_dir, "final.pt"))
    env.close()
    eval_env.close()
    print(f"done in {time.time() - start_time:.0f}s | best eval {best_return:.2f}")


if __name__ == "__main__":
    main()
