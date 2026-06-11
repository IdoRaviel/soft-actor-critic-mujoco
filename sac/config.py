"""Configuration for SAC experiments.

A single ``Config`` dataclass holds every hyperparameter the rest of the code
reads. Per-environment YAML files in ``configs/`` override only the fields that
differ; command-line flags override those in turn. Each run dumps its resolved
config to disk for reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

import yaml


@dataclass
class Config:
    """All knobs for a single SAC training run.

    Defaults follow the SAC paper (arXiv:1812.05905) where applicable; the
    assignment requires us to tune our own values, so per-env YAMLs override.
    """

    # ---- experiment identity ------------------------------------------------
    env_id: str = "Reacher-v4"          # gymnasium MuJoCo v4 environment id
    seed: int = 0                       # RNG seed (python / numpy / torch / env)
    device: str = "auto"                # "auto" | "cpu" | "cuda"
    exp_name: str = "sac"               # label used in output paths

    # ---- training schedule --------------------------------------------------
    total_steps: int = 200_000          # total env.step() calls (the budget)
    start_steps: int = 5_000            # initial steps with random actions; also
                                        #   when gradient updates begin
    updates_per_step: int = 1           # gradient updates per env step
    batch_size: int = 256               # minibatch size sampled from replay
    replay_size: int = 1_000_000        # replay buffer capacity

    # ---- SAC core -----------------------------------------------------------
    gamma: float = 0.99                 # discount factor
    tau: float = 0.005                  # Polyak coefficient for target critics
    lr: float = 3e-4                    # Adam lr (actor, critic, temperature)
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    log_std_min: float = -20.0          # clamp on policy log-std (stability)
    log_std_max: float = 2.0

    # ---- temperature (automatic entropy tuning) -----------------------------
    autotune_alpha: bool = True         # learn alpha vs. target entropy
    init_alpha: float = 0.2             # initial temperature (also used if fixed)
    target_entropy: Optional[float] = None  # None -> -dim(action_space)

    # ---- evaluation ---------------------------------------------------------
    eval_interval: int = 10_000         # evaluate every N env steps (assignment: 10k)
    eval_episodes: int = 10             # episodes averaged per evaluation

    # ---- logging / output ---------------------------------------------------
    out_dir: str = "log"                # root dir for logs/checkpoints/configs
    save_model: bool = True             # save best + final model checkpoints

    # ------------------------------------------------------------------------
    def resolved_device(self) -> str:
        """Turn ``device='auto'`` into a concrete 'cuda'/'cpu' string."""
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def run_name(self) -> str:
        """Unique label for this run, e.g. ``sac_Reacher-v4_seed0``."""
        return f"{self.exp_name}_{self.env_id}_seed{self.seed}"

    def save(self, path: str) -> None:
        """Dump the resolved config to a YAML file for reproducibility."""
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def load_config(path: Optional[str] = None, **overrides) -> Config:
    """Build a ``Config`` from defaults, an optional YAML file, and overrides.

    Precedence (low -> high): dataclass defaults < YAML file < keyword overrides.
    Keyword overrides whose value is ``None`` are ignored, so command-line flags
    that were left unset don't clobber YAML/defaults. Unknown keys raise.
    """
    data: dict = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}

    data.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in fields(Config)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(
            f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(valid)}"
        )
    return Config(**data)
