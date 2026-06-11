"""Plot SAC learning curves from one or more ``eval.csv`` files.

When multiple runs are provided, saves one individual plot per run and one
combined plot with all runs overlaid (plus a dashed mean line).

Examples:
    # three seeds -> 3 individual plots + 1 combined plot
    python scripts/plot.py log/Reacher-v4/seed0_* log/Reacher-v4/seed1_* log/Reacher-v4/seed2_* --out report/reacher.png --title Reacher-v4
    # a single run
    python scripts/plot.py log/Reacher-v4/seed0_20260611-162754 --out report/reacher_seed0.png
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _resolve_csv(path: str) -> str:
    return os.path.join(path, "eval.csv") if os.path.isdir(path) else path


def load_curve(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, means, stds = [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            means.append(float(row["return_mean"]))
            stds.append(float(row["return_std"]))
    return np.array(steps), np.array(means), np.array(stds)


def expand_inputs(inputs: List[str]) -> List[str]:
    csvs: List[str] = []
    for item in inputs:
        matches = glob.glob(item) or [item]
        for m in matches:
            csv_path = _resolve_csv(m)
            if os.path.exists(csv_path):
                csvs.append(csv_path)
    return sorted(set(csvs))


def _make_figure(
    curves: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    no_band: bool,
    show_mean: bool,
) -> plt.Figure:
    """Draw one figure. curves = list of (label, steps, means, stds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    all_means = []
    for label, steps, means, stds in curves:
        line, = ax.plot(steps, means, marker="o", ms=3, lw=1.5, label=label)
        if not no_band:
            ax.fill_between(steps, means - stds, means + stds,
                            color=line.get_color(), alpha=0.15)
        all_means.append((steps, means))

    if show_mean and len(all_means) > 1:
        ref_steps = all_means[0][0]
        if all(np.array_equal(s, ref_steps) for s, _ in all_means):
            stacked = np.stack([m for _, m in all_means])
            ax.plot(ref_steps, stacked.mean(0), color="black", lw=2.5,
                    ls="--", label="mean across runs")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Evaluation return")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot SAC eval learning curves.")
    p.add_argument("inputs", nargs="+", help="Run dirs or eval.csv paths (globs ok).")
    p.add_argument("--out", required=True, help="Output image path for the combined plot.")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--no-band", action="store_true", help="Hide per-run std band.")
    args = p.parse_args()

    csvs = expand_inputs(args.inputs)
    if not csvs:
        raise SystemExit(f"No eval.csv found in: {args.inputs}")

    # Load all curves.
    curves = []
    for csv_path in csvs:
        steps, means, stds = load_curve(csv_path)
        label = os.path.basename(os.path.dirname(csv_path)) or csv_path
        curves.append((label, steps, means, stds))

    base, ext = os.path.splitext(args.out)
    title = args.title or "SAC learning curve"

    # Individual plot per run.
    if len(curves) > 1:
        for i, curve in enumerate(curves):
            fig = _make_figure([curve], f"{title} — {curve[0]}", args.no_band, show_mean=False)
            _save(fig, f"{base}_run{i}{ext}")

    # Combined plot with all runs + mean line.
    fig = _make_figure(curves, title, args.no_band, show_mean=True)
    _save(fig, args.out)


if __name__ == "__main__":
    main()
