"""
guardian.py
===========
Safety Guardian — Slow Layer of Security Cuff.

Process (per paper §3.3):
  1. Encode rollout-prefix observations with frozen RobotObsEncoder.
  2. Run RSSM.observe() to get posterior state sequence from the real prefix.
  3. From the last posterior state, unroll RSSM.img_step() for `rollout_horizon` steps.
  4. Pass each imagined feature through a learned safety head → danger score per step.
  5. Aggregate danger scores → verification score v_t ∈ [0, 1].

Critical synchronization rule:
  Use RSSM.observe() (posterior) over real observations to build history.
  Never feed imagined states back into the history buffer.
  Only img_step() outputs are scored for danger; they are not reused as history.

The RSSM and encoder are FROZEN. Only the safety head is trained.
"""

from __future__ import annotations
import torch
from torch import nn

from models.dreamer_utils import weight_init
from models.rssm import RSSM
from models.encoder import RobotObsEncoder


class SafetyHead(nn.Module):
    """
    MLP that maps a single RSSM feature vector to a scalar danger score.

    Input:  (*, feat_size)
    Output: (*)  in [0, 1]
    """

    def __init__(
        self,
        feat_size: int,
        layers: int = 3,
        units: int = 256,
        act: str = "SiLU",
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        net = []
        d = feat_size
        for _ in range(layers):
            net.append(nn.Linear(d, units))
            net.append(nn.LayerNorm(units, eps=1e-3))
            net.append(act_cls())
            d = units
        net.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*net)
        self.net.apply(weight_init)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feat).squeeze(-1))


class SafetyGuardian(nn.Module):
    """
    Slow-layer consequence-aware verifier.

    Args:
        encoder:          frozen RobotObsEncoder
        rssm:             frozen RSSM
        rollout_horizon:  number of imagined future steps
        feat_size:        RSSM get_feat() output dimension
        safety_head_layers, safety_head_units, safety_head_act:
                          architecture of the learned safety head
        score_aggregation: "mean" or "max" over the imagined horizon
        safety_threshold:  v_t >= threshold → intervene
    """

    def __init__(
        self,
        encoder: RobotObsEncoder,
        rssm: RSSM,
        rollout_horizon: int = 10,
        feat_size: int = 1536,
        safety_head_layers: int = 3,
        safety_head_units: int = 256,
        safety_head_act: str = "SiLU",
        score_aggregation: str = "mean",
        safety_threshold: float = 0.5,
    ):
        super().__init__()

        # Frozen components — do not register as trainable parameters
        self.encoder = encoder
        self.rssm = rssm
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.rssm.parameters():
            p.requires_grad_(False)

        self.rollout_horizon = rollout_horizon
        self.score_aggregation = score_aggregation
        self.safety_threshold = safety_threshold

        # Learned safety head (only this is updated during guardian training)
        self.safety_head = SafetyHead(
            feat_size=feat_size,
            layers=safety_head_layers,
            units=safety_head_units,
            act=safety_head_act,
        )

    def forward(
        self,
        obs_seq: dict,           # {"image": (B,T,H,W,C), "proprio": (B,T,D)}  float32 in [0,1]
        action_seq: torch.Tensor,  # (B, T, action_dim)  robot actions taken during prefix
        is_first: torch.Tensor,    # (B, T)              episode-reset flags
    ) -> torch.Tensor:
        """
        Compute verification score v_t.

        Args:
            obs_seq:    dict of observed image and proprio tensors (B, T, ...)
            action_seq: actions taken during the prefix (B, T, action_dim)
            is_first:   (B, T) bool/float, True at episode start

        Returns:
            v_t: (B,)  verification scores in [0, 1]
        """
        # 1. Encode observations (no grad; encoder is frozen)
        with torch.no_grad():
            embed = self.encoder(obs_seq)                          # (B, T, embed_dim)

        # 2. Observe prefix → posterior states (no grad; rssm is frozen)
        with torch.no_grad():
            post, _ = self.rssm.observe(embed, action_seq, is_first)
            # Take the last posterior state as the starting point for imagination
            last_state = {k: v[:, -1] for k, v in post.items()}   # (B, ...)

        # 3. Roll out imagination (no grad on rssm; grad flows only through safety_head)
        with torch.no_grad():
            imagined = self.rssm.rollout_future(last_state, self.rollout_horizon)
            # imagined["deter"]: (B, horizon, deter)

        # 4. Compute RSSM features for imagined states
        # We need to call get_feat on each time step
        imag_feats = self.rssm.get_feat(imagined)                   # (B, horizon, feat_size)

        # 5. Safety head scores each imagined step (grad flows here)
        danger_scores = self.safety_head(imag_feats)                # (B, horizon)

        # 6. Aggregate over horizon
        if self.score_aggregation == "mean":
            v_t = danger_scores.mean(dim=1)                         # (B,)
        elif self.score_aggregation == "max":
            v_t, _ = danger_scores.max(dim=1)
        else:
            raise NotImplementedError(self.score_aggregation)

        return v_t

    def should_intervene(self, v_t: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where v_t >= safety_threshold."""
        return v_t >= self.safety_threshold

    @torch.no_grad()
    def get_rssm_state_from_prefix(
        self,
        obs_seq: dict,
        action_seq: torch.Tensor,
        is_first: torch.Tensor,
    ) -> dict:
        """
        Extract last posterior RSSM state from a real prefix.
        Called during online inference to maintain the RSSM history.
        """
        embed = self.encoder(obs_seq)
        post, _ = self.rssm.observe(embed, action_seq, is_first)
        return {k: v[:, -1] for k, v in post.items()}

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        encoder: RobotObsEncoder,
        rssm: RSSM,
        feat_size: int,
    ) -> "SafetyGuardian":
        g = cfg["guardian"]
        return cls(
            encoder=encoder,
            rssm=rssm,
            rollout_horizon=g["rollout_horizon"],
            feat_size=feat_size,
            safety_head_layers=g["safety_head_layers"],
            safety_head_units=g["safety_head_units"],
            safety_head_act=g["safety_head_act"],
            score_aggregation=g["score_aggregation"],
            safety_threshold=g["safety_threshold"],
        )


if __name__ == "__main__":
    B, T = 2, 4
    H, W, C = 32, 32, 3
    A = 7
    proprio_dim = 7
    stoch, deter, discrete = 4, 16, 4
    embed_dim = 8 + 16       # tiny CNN out + proprio out

    from models.encoder import RobotObsEncoder
    from models.rssm import RSSM

    enc = RobotObsEncoder(
        image_shape=(H, W, C), cnn_depth=2, minres=8,
        proprio_dim=proprio_dim, proprio_layers=1, proprio_units=16,
    )
    rssm = RSSM(
        stoch=stoch, deter=deter, hidden=deter, discrete=discrete,
        num_actions=A, embed=enc.embed_dim, device="cpu",
    )
    feat_size = stoch * discrete + deter

    guardian = SafetyGuardian(
        encoder=enc, rssm=rssm, rollout_horizon=3,
        feat_size=feat_size,
        safety_head_layers=2, safety_head_units=8,
    )

    obs_seq = {
        "image": torch.rand(B, T, H, W, C),
        "proprio": torch.rand(B, T, proprio_dim),
    }
    action_seq = torch.randn(B, T, A)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    v_t = guardian(obs_seq, action_seq, is_first)
    assert v_t.shape == (B,), v_t.shape
    assert (v_t >= 0).all() and (v_t <= 1).all(), v_t

    # Verify encoder and RSSM are truly frozen
    for p in guardian.encoder.parameters():
        assert not p.requires_grad
    for p in guardian.rssm.parameters():
        assert not p.requires_grad
    # Safety head should be trainable
    assert any(p.requires_grad for p in guardian.safety_head.parameters())

    print(f"guardian: v_t={v_t.tolist()} — smoke checks passed.")
