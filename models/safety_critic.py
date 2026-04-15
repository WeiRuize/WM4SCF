"""
safety_critic.py
================
Safety Critic — maps a frozen RSSM latent feature to a scalar danger score in
[0, 1].  Trained offline on pre-encoded clean + GOBA-poisoned trajectories with
TD(λ) accumulated-cost regression; the Lagrangian wrapper jointly learns a
multiplier λ to enforce an average-cost budget `b`.

Input contract
--------------
`latent_t` is produced by the frozen world model at step t:
    latent_t = cat(z_t_flat, h_t)
where `h_t` is the RSSM deterministic state (deter) and `z_t_flat` is the
flattened stochastic state (stoch * discrete, categorical mode).  The
concatenated dimension is `feat_size`, which must match the RSSM config at
load time; this module does not assume a specific value.

No VLA hidden states, no reference bank, no action context — the world model
is expected to already encode whatever is needed about the dynamics.
"""

from __future__ import annotations
import torch
from torch import nn

from models.dreamer_utils import weight_init


class SafetyCritic(nn.Module):
    """
    Latent → danger-score MLP.

    Architecture (per PROJECT_OVERVIEW §模块二):
        LayerNorm → Linear(feat, units) → SiLU → Dropout
        Linear(units, units)            → SiLU → Dropout
        Linear(units, 1)                → Sigmoid
    """

    def __init__(
        self,
        feat_size: int,
        mlp_units: int = 256,
        mlp_layers: int = 2,
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
            danger: (...,) in [0, 1]
        """
        logit = self.mlp(latent).squeeze(-1)
        return torch.sigmoid(logit)

    @classmethod
    def from_config(cls, cfg: dict, feat_size: int) -> "SafetyCritic":
        c = cfg["critic"]
        return cls(
            feat_size=feat_size,
            mlp_units=c["mlp_units"],
            mlp_layers=c["mlp_layers"],
            dropout=c["dropout"],
            act=c["act"],
        )


class LagrangianSafetyCritic(nn.Module):
    """
    Wraps a SafetyCritic with a learnable log-λ scalar to enforce an average
    cost constraint `E[C] <= b`.

    Training objective (paper §3.2.2):
        L_critic     = MSE(V_c(latent), C_target)
        L_lagrangian = -log_λ.exp() * (C_target.mean() - b).detach()
    The negative sign is because we maximise the dual; gradient descent on
    this term ascends log_λ whenever the constraint is violated.
    """

    def __init__(
        self,
        critic: SafetyCritic,
        budget: float = 0.1,
        lambda_init: float = 1.0,
        lambda_max: float = 100.0,
    ):
        super().__init__()
        self.critic = critic
        self.budget = budget
        self.lambda_max = lambda_max
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(float(lambda_init))))

    @property
    def lambda_value(self) -> torch.Tensor:
        return torch.clamp(self.log_lambda.exp(), max=self.lambda_max)

    def critic_loss(
        self,
        latent: torch.Tensor,          # (B, feat_size)
        cost_target: torch.Tensor,     # (B,) TD(λ) discounted cost
    ) -> torch.Tensor:
        pred = self.critic(latent)
        return torch.nn.functional.mse_loss(pred, cost_target)

    def lagrangian_loss(self, cost_target: torch.Tensor) -> torch.Tensor:
        violation = cost_target.mean().detach() - self.budget
        return -self.lambda_value * violation


if __name__ == "__main__":
    feat_size = 1536
    B = 32
    critic = SafetyCritic(feat_size=feat_size, mlp_units=16, mlp_layers=2, dropout=0.0)
    latent = torch.randn(B, feat_size)
    score = critic(latent)
    assert score.shape == (B,)
    assert (score >= 0).all() and (score <= 1).all()

    lag = LagrangianSafetyCritic(critic, budget=0.1, lambda_init=1.0)
    cost = torch.rand(B)
    l_c = lag.critic_loss(latent, cost)
    l_lag = lag.lagrangian_loss(cost)
    assert l_c.ndim == 0 and l_lag.ndim == 0
    print(f"safety_critic: score.mean={score.mean():.3f}, "
          f"L_critic={l_c.item():.4f}, L_lag={l_lag.item():.4f} — smoke ok.")
