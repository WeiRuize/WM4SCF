"""
safety_critic.py
================
Safety Critic — maps a frozen RSSM latent feature to a predicted future
intrinsic-cost λ-return.

Training target is NOT a binary danger label; it is the discounted sum of
per-step world-model surprises (KL between one-step posterior and imagined
prior) along an imagination rollout seeded from real demonstration data.
See train/train_critic.py for the procedure.

Output contract:
    critic(latent) → non-negative scalar ≈ E[Σ γᵏ · kl_{t+k+1}]

We use softplus rather than sigmoid because the target is an unbounded
non-negative quantity, not a probability.  The online Safety Guardian
compares this value to a threshold calibrated from the clean-validation
histogram (see docs §online-inference).
"""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from models.dreamer_utils import weight_init


class SafetyCritic(nn.Module):
    """
    Latent → predicted future intrinsic cost.

    Architecture:
        LayerNorm(feat) → [Linear → act → Dropout] × mlp_layers → Linear(→1)
        final activation: softplus (cost ≥ 0)
    """

    def __init__(
        self,
        feat_size: int,
        mlp_units: int = 512,
        mlp_layers: int = 3,
        dropout: float = 0.1,
        act: str = "SiLU",
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        layers: list[nn.Module] = [nn.LayerNorm(feat_size, eps=1e-3)]
        d = feat_size
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d, mlp_units))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = mlp_units
        layers.append(nn.Linear(mlp_units, 1))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(weight_init)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (..., feat_size)
        Returns:
            cost:   (...,) ≥ 0
        """
        raw = self.mlp(latent).squeeze(-1)
        return F.softplus(raw)

    @classmethod
    def from_config(cls, cfg: dict, feat_size: int) -> "SafetyCritic":
        c = cfg["critic"]
        return cls(
            feat_size=feat_size,
            mlp_units=c["mlp_units"],
            mlp_layers=c["mlp_layers"],
            dropout=c.get("dropout", 0.1),
            act=c.get("act", "SiLU"),
        )


if __name__ == "__main__":
    feat_size = 1536
    B, T = 4, 16
    critic = SafetyCritic(feat_size=feat_size, mlp_units=64, mlp_layers=2, dropout=0.0)

    latent = torch.randn(B, T, feat_size)
    pred = critic(latent)
    assert pred.shape == (B, T)
    assert (pred >= 0).all()
    print(f"safety_critic: pred.shape={tuple(pred.shape)}, "
          f"pred.mean={pred.mean():.3f} — smoke ok.")
