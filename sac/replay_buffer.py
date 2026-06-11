"""Fixed-size FIFO experience replay buffer for off-policy SAC.

Backed by pre-allocated numpy arrays for a bounded memory footprint and fast
vectorized minibatch sampling. ``sample`` returns torch tensors on the target
device, ready for the agent's update step.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    """Circular buffer of (state, action, reward, next_state, done) transitions.

    ``done`` should carry the environment's ``terminated`` flag only (true
    terminal states), never ``truncated`` (time-limit cutoffs), so that value
    is still bootstrapped past artificial episode truncations.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0    # next write index
        self.size = 0   # number of valid transitions stored

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest when full."""
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random minibatch as torch tensors on ``self.device``."""
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda arr: torch.as_tensor(arr[idx], device=self.device)
        return {
            "states": to_t(self.states),
            "actions": to_t(self.actions),
            "rewards": to_t(self.rewards),
            "next_states": to_t(self.next_states),
            "dones": to_t(self.dones),
        }

    def __len__(self) -> int:
        return self.size
