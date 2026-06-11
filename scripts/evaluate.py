"""Evaluate trained SAC checkpoints and optionally record a video.

Accepts one or more run directories. When multiple runs are given, scores each
one and prints a combined summary (individual scores + average).

Examples:
    # evaluate 3 seeds in one command
    python scripts/evaluate.py --runs log/Reacher-v4/seed0_* log/Reacher-v4/seed1_* log/Reacher-v4/seed2_* --episodes 200

    # single run + record a video
    python scripts/evaluate.py --runs log/Reacher-v4/seed0_<timestamp> --episodes 200 --video videos/reacher.mp4
"""

from __future__ import annotations

import argparse
import glob
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from sac.agent import SACAgent
from sac.config import load_config
from sac.utils import set_seed


def build_agent(cfg, env: gym.Env) -> SACAgent:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return SACAgent(obs_dim, act_dim, env.action_space.low, env.action_space.high, cfg)


def score(agent: SACAgent, env: gym.Env, episodes: int, seed_base: int) -> tuple:
    """Return (mean, std) of deterministic-policy returns over ``episodes``."""
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


def record(agent: SACAgent, env_id: str, path: str, episodes: int, seed_base: int) -> None:
    """Render ``episodes`` deterministic episodes and write them to one mp4."""
    env = gym.make(env_id, render_mode="rgb_array")
    fps = env.metadata.get("render_fps", 30)
    frames = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed_base + ep)
        done = False
        while not done:
            frames.append(env.render())
            action = agent.select_action(state, evaluate=True)
            state, _, term, trunc, _ = env.step(action)
            done = term or trunc
    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    print(f"saved {path} ({len(frames)} frames @ {fps} fps)")


def expand_runs(inputs: list[str]) -> list[str]:
    """Expand globs into a sorted list of existing run directories."""
    runs = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for m in matches:
            if os.path.isdir(m) and os.path.exists(os.path.join(m, "config.yaml")):
                runs.append(m)
    return sorted(set(runs))


def format_results(
    env_id: str,
    results: list[tuple[str, float, float]],
    episodes: int,
    checkpoint: str,
) -> str:
    """Format evaluation results as a human-readable string."""
    sep = "-" * 42
    lines = [
        f"Environment : {env_id}",
        f"Checkpoint  : {checkpoint}",
        f"Episodes    : {episodes}",
        sep,
    ]
    means = []
    for label, mean, std in results:
        lines.append(f"  {label}: {mean:.2f} +/- {std:.2f}")
        means.append(mean)
    if len(means) > 1:
        lines.append(sep)
        lines.append(f"  runs:    {[round(m, 2) for m in means]}")
        lines.append(f"  average: {float(np.mean(means)):.2f}")
    return "\n".join(lines)


def save_results(text: str, env_id: str) -> None:
    """Save results to report/<env_id>/eval_results.txt."""
    out_path = os.path.join("report", env_id, "eval_results.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nsaved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate / record SAC checkpoints.")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run dirs with config.yaml + checkpoint (globs ok).")
    p.add_argument("--checkpoint", default="best.pt", help="Checkpoint filename in each run dir.")
    p.add_argument("--episodes", type=int, default=200, help="Episodes to score per run.")
    p.add_argument("--seed", type=int, default=20_000, help="Eval seed base.")
    p.add_argument("--video", default=None, help="Output mp4 path (records the first run only).")
    p.add_argument("--video-episodes", type=int, default=2, help="Episodes to record.")
    args = p.parse_args()

    runs = expand_runs(args.runs)
    if not runs:
        raise SystemExit(f"No valid run directories found in: {args.runs}")

    results = []
    env_id = None
    for run in runs:
        cfg = load_config(os.path.join(run, "config.yaml"))
        env_id = cfg.env_id
        set_seed(cfg.seed)
        env = gym.make(cfg.env_id)
        agent = build_agent(cfg, env)
        agent.load(os.path.join(run, args.checkpoint))
        mean, std = score(agent, env, args.episodes, args.seed)
        env.close()
        label = os.path.basename(run)
        results.append((label, mean, std))
        print(f"  {label}: {mean:.2f} +/- {std:.2f}")

    text = format_results(env_id, results, args.episodes, args.checkpoint)
    print(f"\n{text}")
    save_results(text, env_id)

    if args.video:
        cfg = load_config(os.path.join(runs[0], "config.yaml"))
        env = gym.make(cfg.env_id)
        agent = build_agent(cfg, env)
        agent.load(os.path.join(runs[0], args.checkpoint))
        env.close()
        record(agent, cfg.env_id, args.video, args.video_episodes, args.seed)


if __name__ == "__main__":
    main()
