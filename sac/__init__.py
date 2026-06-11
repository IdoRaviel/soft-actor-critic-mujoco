"""From-scratch Soft Actor-Critic (SAC) implementation in PyTorch.

Implements the algorithm from Haarnoja et al., 2018/2019,
"Soft Actor-Critic Algorithms and Applications" (arXiv:1812.05905):
twin Q-critics with target networks, a tanh-squashed Gaussian policy, and
automatic entropy-temperature tuning. No external RL libraries are used.
"""

from sac.config import Config, load_config

__all__ = ["Config", "load_config"]
