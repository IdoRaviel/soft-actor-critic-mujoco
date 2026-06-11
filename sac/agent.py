"""The Soft Actor-Critic agent (arXiv:1812.05905).

Owns the policy, twin critics + their targets, and the (optionally learned)
entropy temperature. ``update`` performs one gradient step of the algorithm:
critic -> actor -> temperature -> Polyak target update. Environment interaction
goes through ``select_action``; ``save``/``load`` handle checkpoints.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from sac.config import Config
from sac.networks import GaussianPolicy, QNetwork


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Polyak averaging: target <- tau*source + (1-tau)*target (in place)."""
    for s, t in zip(source.parameters(), target.parameters()):
        t.data.mul_(1.0 - tau).add_(tau * s.data)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        cfg: Config,
    ):
        self.cfg = cfg
        self.device = torch.device(cfg.resolved_device())
        self.gamma = cfg.gamma
        self.tau = cfg.tau

        # --- policy and twin critics (+ frozen target critics) --------------
        self.policy = GaussianPolicy(
            obs_dim, act_dim, cfg.hidden_sizes, action_low, action_high,
            cfg.log_std_min, cfg.log_std_max,
        ).to(self.device)

        self.q1 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q2 = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in (*self.q1_target.parameters(), *self.q2_target.parameters()):
            p.requires_grad_(False)  # targets are updated by Polyak, not gradients

        # --- optimizers -----------------------------------------------------
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr
        )

        # --- entropy temperature alpha --------------------------------------
        self.autotune = cfg.autotune_alpha
        if self.autotune:
            # Target entropy defaults to -dim(action) (paper's heuristic).
            self.target_entropy = (
                float(cfg.target_entropy)
                if cfg.target_entropy is not None
                else -float(act_dim)
            )
            # Optimize log_alpha (keeps alpha > 0) rather than alpha directly.
            self.log_alpha = torch.tensor(
                np.log(cfg.init_alpha), dtype=torch.float32,
                device=self.device, requires_grad=True,
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            # Fixed temperature.
            self.log_alpha = torch.tensor(
                np.log(cfg.init_alpha), dtype=torch.float32, device=self.device
            )

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Pick an action for a single observation (returns a numpy array)."""
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if evaluate:
            action = self.policy.deterministic_action(state_t)
        else:
            action, _ = self.policy.sample(state_t)
        return action.squeeze(0).cpu().numpy()

    # -----------------------------------------------------------------------
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """One SAC gradient step. Returns scalar metrics for logging."""
        s, a = batch["states"], batch["actions"]
        r, s2, d = batch["rewards"], batch["next_states"], batch["dones"]

        # --- (1) critic update ---------------------------------------------
        with torch.no_grad():
            a2, logp2 = self.policy.sample(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha * logp2
            target = r + self.gamma * (1.0 - d) * min_q_t

        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- (2) actor update ----------------------------------------------
        a_pi, logp = self.policy.sample(s)
        q1_pi = self.q1(s, a_pi)
        q2_pi = self.q2(s, a_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- (3) temperature update ----------------------------------------
        if self.autotune:
            # Drive entropy (-logp) toward the target entropy.
            alpha_loss = -(
                self.log_alpha * (logp + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        # --- (4) Polyak update of target critics ---------------------------
        soft_update(self.q1, self.q1_target, self.tau)
        soft_update(self.q2, self.q2_target, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": self.alpha.item(),
            "entropy": (-logp).mean().item(),
            "q_mean": min_q_pi.mean().item(),
        }

    # -----------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
