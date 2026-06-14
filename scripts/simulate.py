"""Watch a trained SAC agent act live in the MuJoCo viewer.

Unlike ``evaluate.py`` (which scores and records an mp4), this opens a real-time
window (``render_mode="human"``) so you can watch the policy. Pass ``--gif`` to
record to a GIF file instead (no live window; works on headless servers with
MUJOCO_GL=egl).

Examples:
    python scripts/simulate.py --checkpoint log/sac_Reacher-v4_seed0_20260611-154530/best.pt
    python scripts/simulate.py --checkpoint log/sac_Pusher-v4_seed0_.../best.pt --gif videos/pusher.gif --gif-episodes 3
"""

from __future__ import annotations

import argparse
import os
import random

import gymnasium as gym
import imageio.v2 as imageio

from sac.agent import SACAgent
from sac.config import load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Render a SAC checkpoint live or save to GIF.")
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    p.add_argument("--config", default=None,
                   help="Path to config.yaml (default: alongside the checkpoint).")
    p.add_argument("--gif", default=None, help="Save a GIF to this path instead of live render.")
    p.add_argument("--gif-episodes", type=int, default=1, help="Episodes to record into the GIF.")
    p.add_argument("--gif-max-steps", type=int, default=300, help="Max steps to record per episode.")
    args = p.parse_args()

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = args.config or os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(
            f"config.yaml not found at {config_path}; pass --config explicitly."
        )

    cfg = load_config(config_path)

    render_mode = "rgb_array" if args.gif else "human"
    env = gym.make(cfg.env_id, render_mode=render_mode)
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        cfg,
    )
    agent.load(args.checkpoint)

    if args.gif:
        fps = env.metadata.get("render_fps", 30)
        frames = []
        print(f"Recording {args.gif_episodes} episode(s) to {args.gif} (max {args.gif_max_steps} steps each) ...")
        for ep in range(args.gif_episodes):
            state, _ = env.reset(seed=random.randint(0, 2**31))
            done, total, step = False, 0.0, 0
            while not done and step < args.gif_max_steps:
                frames.append(env.render())
                action = agent.select_action(state, evaluate=True)
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
                step += 1
            print(f"  episode {ep + 1}: {step} steps, return {total:.2f}")
        env.close()
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        imageio.mimsave(args.gif, frames, fps=fps)
        print(f"saved {args.gif} ({len(frames)} frames @ {fps} fps)")
    else:
        print(f"Simulating {cfg.env_id} | {args.checkpoint} | running until Ctrl+C")
        ep = 0
        try:
            while True:
                state, _ = env.reset(seed=random.randint(0, 2**31))
                done, total = False, 0.0
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    state, reward, term, trunc, _ = env.step(action)
                    total += reward
                    done = term or trunc
                ep += 1
                print(f"  episode {ep}: return {total:.2f}")
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            env.close()


if __name__ == "__main__":
    main()
