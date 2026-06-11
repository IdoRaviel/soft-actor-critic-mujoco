"""Small utilities: reproducible seeding, CSV logging, run directories."""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(out_dir: str, run_name: str) -> str:
    """Create and return ``<out_dir>/<run_name>/`` for this run's artifacts."""
    path = os.path.join(out_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return path


class CSVLogger:
    """Append-only CSV writer with a fixed set of columns."""

    def __init__(self, path: str, fieldnames: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: Dict[str, float]) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
