"""
safety/guardian.py
==================
Online Safety Guardian — runs between the VLA and the robot executor.

Two inference entry points:

    guardian.evaluate_action(obs_t, action_t)
        # OpenVLA single-step path (Experiments A, B).

    guardian.evaluate_chunk(obs_t, action_chunk)
        # π₀ multi-step path (Experiments C, D).

Both maintain the frozen RSSM history `(h_prev, z_prev)` across steps.  Per
PROJECT_OVERVIEW §原则七, the history is ONLY updated with the real posterior
state computed from `obs_t`; imagined states are used exclusively for danger
scoring.

This module is pure inference: the RSSM, encoder, and SafetyCritic must all
be loaded from checkpoints and frozen before being handed in.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from models.safety_critic import SafetyCritic


@dataclass
class GuardianDecision:
    """Return value of evaluate_action / evaluate_chunk."""
    block: bool
    danger_score: float                      # aggregated score used for the decision
    per_step_scores: Optional[list[float]]   # per-step imagined danger (chunk mode)
    safe_actions: torch.Tensor               # actions the guardian allows through
    first_danger_step: Optional[int]         # index of first over-threshold step, if any


class SafetyGuardian:
    def __init__(
        self,
        encoder: RobotObsEncoder,
        rssm: RSSM,
        critic: SafetyCritic,
        safety_threshold: float = 0.5,
        gamma: float = 0.9,
        device: str = "cuda:0",
    ):
        self.encoder = encoder.to(device).eval()
        self.rssm = rssm.to(device).eval()
        self.critic = critic.to(device).eval()
        for m in (self.encoder, self.rssm, self.critic):
            for p in m.parameters():
                p.requires_grad_(False)

        self.threshold = safety_threshold
        self.gamma = gamma
        self.device = device
        self._state: Optional[dict] = None   # last real posterior RSSM state

    # ── Episode lifecycle ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Call at the start of every new episode (after env.reset())."""
        self._state = None

    # ── Core inference ────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_and_advance(self, obs_t: dict) -> dict:
        """
        Encode the real observation and advance the RSSM posterior.  Updates
        and returns `self._state`.  Creates the initial state lazily on the
        first call of each episode.
        """
        image = obs_t["image"].to(self.device).float() / 255.0
        proprio = obs_t["proprio"].to(self.device).float()
        image = image.unsqueeze(0).unsqueeze(0)        # (1, 1, H, W, C)
        proprio = proprio.unsqueeze(0).unsqueeze(0)    # (1, 1, D)
        embed = self.encoder({"image": image, "proprio": proprio})    # (1, 1, E)

        is_first = torch.zeros(1, 1, device=self.device)
        if self._state is None:
            is_first[0, 0] = 1.0
        action_prev = torch.zeros(1, 1, self.rssm._num_actions, device=self.device)

        post, _ = self.rssm.observe(embed, action_prev, is_first, state=self._state)
        self._state = {k: v[:, -1] for k, v in post.items()}   # (1, ...)
        return self._state

    @torch.no_grad()
    def evaluate_action(self, obs_t: dict, action_t: torch.Tensor) -> GuardianDecision:
        """OpenVLA single-step verification."""
        state = self._encode_and_advance(obs_t)
        action = action_t.to(self.device).float().reshape(1, -1)

        # Imagine one step under the proposed action
        next_state = self.rssm.img_step(state, action, sample=False)
        feat = self.rssm.get_feat(next_state)                 # (1, feat)
        score = float(self.critic(feat).item())

        block = score >= self.threshold
        return GuardianDecision(
            block=block,
            danger_score=score,
            per_step_scores=[score],
            safe_actions=action.detach().cpu() if not block else action.detach().cpu()[:0],
            first_danger_step=0 if block else None,
        )

    @torch.no_grad()
    def evaluate_chunk(self, obs_t: dict, action_chunk: torch.Tensor) -> GuardianDecision:
        """
        π₀ chunk verification.

        Args:
            action_chunk: (H, action_dim) sequence of future actions.
        """
        state = self._encode_and_advance(obs_t)
        chunk = action_chunk.to(self.device).float()
        H = chunk.shape[0]

        per_step: list[float] = []
        cumulative = 0.0
        first_danger: Optional[int] = None
        rolling = state
        for k in range(H):
            rolling = self.rssm.img_step(rolling, chunk[k:k + 1], sample=False)
            feat = self.rssm.get_feat(rolling)
            s_k = float(self.critic(feat).item())
            per_step.append(s_k)
            cumulative += (self.gamma ** k) * s_k
            if first_danger is None and s_k >= self.threshold:
                first_danger = k

        if first_danger is None:
            return GuardianDecision(
                block=False,
                danger_score=cumulative,
                per_step_scores=per_step,
                safe_actions=chunk.detach().cpu(),
                first_danger_step=None,
            )

        # BLOCK: execute only the prefix up to the first over-threshold step
        return GuardianDecision(
            block=True,
            danger_score=cumulative,
            per_step_scores=per_step,
            safe_actions=chunk[:first_danger].detach().cpu(),
            first_danger_step=first_danger,
        )

    @classmethod
    def from_checkpoints(
        cls,
        cfg: dict,
        rssm_ckpt: str,
        critic_ckpt: str,
        device: str = "cuda:0",
    ) -> "SafetyGuardian":
        """Reconstruct encoder + RSSM + critic from config and checkpoints."""
        encoder = RobotObsEncoder.from_config(cfg)
        rssm = RSSM(
            stoch=cfg["rssm"]["stoch"],
            deter=cfg["rssm"]["deter"],
            hidden=cfg["rssm"]["hidden"],
            discrete=cfg["rssm"]["discrete"],
            num_actions=cfg["rssm"]["num_actions"],
            embed=encoder.embed_dim,
            device=device,
        )
        feat_size = rssm._stoch * max(rssm._discrete, 1) + rssm._deter
        critic = SafetyCritic.from_config(cfg, feat_size=feat_size)

        rssm_state = torch.load(rssm_ckpt, map_location=device)
        encoder.load_state_dict(rssm_state["encoder"])
        rssm.load_state_dict(rssm_state["rssm"])
        critic.load_state_dict(torch.load(critic_ckpt, map_location=device)["critic"])

        return cls(
            encoder=encoder,
            rssm=rssm,
            critic=critic,
            safety_threshold=cfg["eval"]["safety_threshold"],
            gamma=cfg["eval"].get("gamma", 0.9),
            device=device,
        )
