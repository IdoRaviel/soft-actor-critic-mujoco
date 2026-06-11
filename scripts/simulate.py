"""Watch a trained SAC agent act live in the MuJoCo viewer.

Unlike ``evaluate.py`` (which scores and records an mp4), this opens a real-time
window (``render_mode="human"``) so you can watch the policy. The environment
and network architecture are read from the run's ``config.yaml``, which is
located next to the checkpoint by default.

Examples:
    python scripts/simulate.py --checkpoint log/sac_Reacher-v4_seed0_20260611-154530/best.pt
    python scripts/simulate.py --checkpoint log/sac_Pusher-v4_seed0_.../final.pt --episodes 5
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym

from sac.agent import SACAgent
from sac.config import load_config
from sac.utils import set_seed


def main() -> None:
    p = argparse.ArgumentParser(description="Render a SAC checkpoint live.")
    p.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    p.add_argument("--config", default=None,
                   help="Path to config.yaml (default: alongside the checkpoint).")
    p.add_argument("--episodes", type=int, default=5, help="Episodes to play.")
    p.add_argument("--seed", type=int, default=20_000, help="Eval seed base.")
    args = p.parse_args()

    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = args.config or os.path.join(ckpt_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(
            f"config.yaml not found at {config_path}; pass --config explicitly."
        )

    cfg = load_config(config_path)
    set_seed(cfg.seed)

    # Live viewer. Force CPU: a single rendered env gains nothing from the GPU.
    env = gym.make(cfg.env_id, render_mode="human")
    agent = SACAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
        cfg,
    )
    agent.load(args.checkpoint)

    print(f"Simulating {cfg.env_id} | {args.checkpoint} | {args.episodes} episode(s) "
          f"(close the window or Ctrl+C to stop)")
    try:
        for ep in range(args.episodes):
            state, _ = env.reset(seed=args.seed + ep)
            done, total = False, 0.0
            while not done:
                action = agent.select_action(state, evaluate=True)
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
            print(f"  episode {ep + 1}: return {total:.2f}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
