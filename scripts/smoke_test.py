"""
scripts/smoke_test.py
=====================
End-to-end wiring test that runs in under two minutes on CPU.

What it checks:
    1. encoder → rssm → decoder forward pass on a tiny random batch.
    2. SafetyCritic + LagrangianSafetyCritic forward and loss.
    3. SafetyGuardian single-step and chunk paths with frozen modules.

It never reads real data, never loads real checkpoints, and never touches
any VLA.  Use it as a fast sanity check before running阶段一 training.
"""

from __future__ import annotations
import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from models.decoder import RobotObsDecoder
from models.safety_critic import SafetyCritic, LagrangianSafetyCritic
from safety.guardian import SafetyGuardian


def main():
    device = "cpu"
    B, T = 2, 6
    H = W = 32
    stoch, deter, discrete = 4, 16, 4
    action_dim = 7

    enc = RobotObsEncoder(
        image_shape=(H, W, 3), cnn_depth=4, minres=4,
        proprio_dim=action_dim, proprio_layers=1, proprio_units=16,
    )
    rssm = RSSM(
        stoch=stoch, deter=deter, hidden=deter, discrete=discrete,
        num_actions=action_dim, embed=enc.embed_dim, device=device,
    )
    feat_size = stoch * discrete + deter
    dec = RobotObsDecoder(
        feat_size=feat_size, image_shape=(H, W, 3),
        cnn_depth=4, minres=4, proprio_dim=action_dim,
        mlp_layers=1, mlp_units=16,
    )
    critic = SafetyCritic(feat_size=feat_size, mlp_units=16, mlp_layers=2, dropout=0.0)
    lag = LagrangianSafetyCritic(critic, budget=0.1, lambda_init=1.0)

    # 1. World-model forward
    obs = {
        "image": torch.rand(B, T, H, W, 3),
        "proprio": torch.rand(B, T, action_dim),
    }
    action = torch.randn(B, T, action_dim)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0
    embed = enc(obs)
    post, prior = rssm.observe(embed, action, is_first)
    feat = rssm.get_feat(post)
    assert feat.shape == (B, T, feat_size)
    dists = dec(feat)
    assert dists["image"].mode().shape == (B, T, H, W, 3)

    # 2. Critic + Lagrangian loss
    flat = feat.reshape(-1, feat_size)
    score = critic(flat)
    assert score.shape == (B * T,)
    cost = torch.rand(B * T)
    l_c = lag.critic_loss(flat, cost)
    l_l = lag.lagrangian_loss(cost)
    l_c.backward()
    l_l.backward()

    # 3. Guardian single-step + chunk
    guardian = SafetyGuardian(
        encoder=enc, rssm=rssm, critic=critic,
        safety_threshold=0.5, gamma=0.9, device=device,
    )
    frame = {"image": torch.rand(H, W, 3) * 255, "proprio": torch.rand(action_dim)}
    d_step = guardian.evaluate_action(frame, torch.zeros(action_dim))
    assert 0.0 <= d_step.danger_score <= 1.0
    d_chunk = guardian.evaluate_chunk(frame, torch.zeros(5, action_dim))
    assert len(d_chunk.per_step_scores) == 5

    print("smoke_test: encoder/rssm/decoder/critic/guardian all wired — OK.")


if __name__ == "__main__":
    main()
