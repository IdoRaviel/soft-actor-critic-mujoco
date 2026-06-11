"""Neural networks for SAC: a tanh-squashed Gaussian policy and Q-critics.

Architecture follows arXiv:1812.05905 (2 hidden layers of 256 units, ReLU).
The value network from the first SAC paper is intentionally absent: this
version bootstraps the target directly from the twin target critics.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Numerical floor used inside the tanh log-prob correction.
LOG_PROB_EPS = 1e-6


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int) -> nn.Sequential:
    """Build a ReLU MLP: in_dim -> hidden_sizes -> out_dim (no output activation)."""
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """A single soft Q-function Q(s, a) -> scalar.

    The agent instantiates two of these (plus two target copies) and takes the
    elementwise minimum to reduce overestimation bias (clipped double-Q).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)  # shape (batch, 1)


class GaussianPolicy(nn.Module):
    """Tanh-squashed diagonal-Gaussian policy.

    Outputs a state-dependent mean and log-std, samples with the
    reparameterization trick, squashes through tanh, and rescales to the
    environment's action bounds. ``sample`` returns the action together with the
    corrected log-probability needed by SAC's entropy term.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk, then separate mean / log-std heads.
        self.trunk = mlp(obs_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.trunk.append(nn.ReLU())  # activate the last trunk layer
        self.mean_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)

        # Affine map from tanh-space [-1, 1] to the env's action range.
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the pre-tanh Gaussian."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action (reparameterized) and its corrected log-prob.

        Returns (action, log_prob) with action in the env range and log_prob of
        shape (batch, 1).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        # A Gaussian has unbounded range (-inf, +inf) but the environment only
        # accepts actions within [low, high]. We can't send the raw sample to
        # the env. Solution: squash through tanh to guarantee [-1, 1], then
        # rescale to the env's range with the affine map learned at __init__.
        #   u ~ Gaussian (unbounded)
        #   y = tanh(u)              -> y in [-1, 1]
        #   action = y*scale + bias  -> action in [low, high]
        u = normal.rsample()                # reparameterized pre-tanh sample
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias

        # log_prob in pre-tanh space, then correct for tanh + affine squashing.
        log_prob = normal.log_prob(u)
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + LOG_PROB_EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """The distribution's mode: tanh(mean) rescaled to the action range.

        Used for evaluation. This is pure distribution logic (no sampling); the
        agent wraps it under ``no_grad`` when stepping the environment.
        """
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias
