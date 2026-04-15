"""
critic.py
=========
Safety Critic — Fast Layer of Security Cuff.

Implements:
  - SafetyCritic: MLP that maps φ_t = [h_t_proj; δ_t; u_t] → s_t ∈ [0,1]
  - RiskAggregator: maintains online rollout-level risk r_t via EMA

Architecture (per paper §3.2):
  h_t       = VLA LLM last-layer hidden state  (4096D from LLaMA-2 7B)
  h_t_proj  = Linear(4096 → proj_dim=512)
  δ_t       = h_t_proj − ρ_t(S)     residual from success reference bank
  u_t       = cat(recent_actions)    context_horizon × action_dim = 35D
  φ_t       = [h_t_proj; δ_t; u_t]  → 1059D → MLP → sigmoid → s_t

Online aggregation:
  r_t = α·s_t + (1−α)·r_{t−1}       (EMA with α = ema_alpha)
  if r_t ≥ γ  → escalate to slow layer
"""

from __future__ import annotations
import torch
from torch import nn
from models.dreamer_utils import weight_init


class SafetyCritic(nn.Module):
    """
    Fast-layer MLP risk scorer.

    Args (from configs/critic.yaml):
        vla_hidden_dim:   LLM hidden dim to project from
        proj_dim:         projected dimension (also ref-bank feature dim)
        action_dim:       robot action dimension
        context_horizon:  number of recent steps of actions to include
        mlp_layers:       MLP depth
        mlp_units:        MLP width
        dropout:          dropout probability
        act:              activation class name (e.g. "SiLU")
    """

    def __init__(
        self,
        vla_hidden_dim: int = 4096,
        proj_dim: int = 512,
        action_dim: int = 7,
        context_horizon: int = 5,
        mlp_layers: int = 4,
        mlp_units: int = 256,
        dropout: float = 0.1,
        act: str = "SiLU",
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.context_horizon = context_horizon
        self.action_dim = action_dim

        # Project VLA hidden state to manageable dim
        self.projector = nn.Linear(vla_hidden_dim, proj_dim)

        # MLP classifier
        # input: [h_proj; delta; u_t] = proj_dim + proj_dim + context_horizon*action_dim
        in_dim = proj_dim + proj_dim + context_horizon * action_dim
        act_cls = getattr(nn, act)
        layers = []
        d = in_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(d, mlp_units))
            layers.append(nn.LayerNorm(mlp_units, eps=1e-3))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = mlp_units
        layers.append(nn.Linear(mlp_units, 1))
        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(weight_init)
        self.projector.apply(weight_init)

    def project(self, vla_hidden: torch.Tensor) -> torch.Tensor:
        """
        Project raw VLA hidden state.

        Args:
            vla_hidden: (B, vla_hidden_dim) or (B, T, vla_hidden_dim)
        Returns:
            h_proj: same leading dims but last dim = proj_dim
        """
        return self.projector(vla_hidden)

    def forward(
        self,
        vla_hidden: torch.Tensor,       # (B, vla_hidden_dim)
        reference_mean: torch.Tensor,   # (B, proj_dim) — ρ_t(S) from reference bank
        recent_actions: torch.Tensor,   # (B, context_horizon, action_dim)
    ) -> torch.Tensor:
        """
        Compute per-step risk score s_t.

        Returns:
            s_t: (B,) risk scores in [0, 1]
        """
        B = vla_hidden.shape[0]
        h_proj = self.projector(vla_hidden)                       # (B, proj_dim)
        delta = h_proj - reference_mean                           # (B, proj_dim)
        u_t = recent_actions.reshape(B, -1)                       # (B, context_horizon*action_dim)
        phi = torch.cat([h_proj, delta, u_t], dim=-1)             # (B, in_dim)
        logit = self.mlp(phi).squeeze(-1)                         # (B,)
        return torch.sigmoid(logit)

    @classmethod
    def from_config(cls, cfg: dict) -> "SafetyCritic":
        c = cfg["critic"]
        return cls(
            vla_hidden_dim=c["vla_hidden_dim"],
            proj_dim=c["proj_dim"],
            action_dim=c["action_dim"],
            context_horizon=c["context_horizon"],
            mlp_layers=c["mlp_layers"],
            mlp_units=c["mlp_units"],
            dropout=c["dropout"],
            act=c["act"],
        )


class RiskAggregator:
    """
    Online rollout-level risk estimate via EMA.

    Usage:
        agg = RiskAggregator(ema_alpha=0.3, escalation_threshold=0.65)
        for t, s_t in enumerate(step_scores):
            r_t = agg.update(s_t)
            if agg.should_escalate():
                run_slow_layer()
                agg.reset()
    """

    def __init__(self, ema_alpha: float = 0.3, escalation_threshold: float = 0.65):
        self.alpha = ema_alpha
        self.threshold = escalation_threshold
        self.r_t: float = 0.0

    def update(self, s_t: float) -> float:
        """Update EMA risk and return new r_t."""
        self.r_t = self.alpha * s_t + (1.0 - self.alpha) * self.r_t
        return self.r_t

    def should_escalate(self) -> bool:
        return self.r_t >= self.threshold

    def reset(self):
        """Reset at start of each new rollout."""
        self.r_t = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskAggregator":
        d = cfg["detector"]
        return cls(
            ema_alpha=d["ema_alpha"],
            escalation_threshold=d["escalation_threshold"],
        )


if __name__ == "__main__":
    import torch

    B = 4
    vla_dim, proj_dim, action_dim, horizon = 16, 8, 7, 5

    critic = SafetyCritic(
        vla_hidden_dim=vla_dim,
        proj_dim=proj_dim,
        action_dim=action_dim,
        context_horizon=horizon,
        mlp_layers=2,
        mlp_units=16,
        dropout=0.0,
    )

    vla_h = torch.randn(B, vla_dim)
    ref = torch.randn(B, proj_dim)
    actions = torch.randn(B, horizon, action_dim)

    scores = critic(vla_h, ref, actions)
    assert scores.shape == (B,), scores.shape
    assert (scores >= 0).all() and (scores <= 1).all(), scores

    # Test training forward with batched time dim
    T = 6
    vla_h_t = torch.randn(B, T, vla_dim)
    ref_t = torch.randn(B, T, proj_dim)
    actions_t = torch.randn(B, T, horizon, action_dim)
    # Flatten B*T for batch training
    scores_t = critic(
        vla_h_t.reshape(B * T, vla_dim),
        ref_t.reshape(B * T, proj_dim),
        actions_t.reshape(B * T, horizon, action_dim),
    )
    assert scores_t.shape == (B * T,), scores_t.shape

    # RiskAggregator
    agg = RiskAggregator(ema_alpha=0.3, escalation_threshold=0.65)
    for step_score in [0.1, 0.2, 0.8, 0.9]:
        r = agg.update(step_score)
        assert 0.0 <= r <= 1.0

    print("critic: all smoke checks passed.")
