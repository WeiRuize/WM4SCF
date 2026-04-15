"""
train_critic.py
===============
Stage-2: self-supervised Safety Critic training.

Philosophy
----------
The RSSM world model (Stage 1) was trained purely on CLEAN LIBERO via ELBO
and is a density estimator for the clean support.  Its per-step KL between
posterior (after seeing the real observation) and prior (the one-step
imagined belief) is the built-in "surprise" signal: low on clean demos,
high on GOBA-poisoned demos that the world model has never seen.

We exploit this to train the Safety Critic WITHOUT any backdoor labels, in
the Dreamer-V3 spirit:

    1.  Sample a window (image, proprio, action) of length T from a mix of
        clean and poisoned HDF5 files.
    2.  Frozen encoder + RSSM encode the full window into real posteriors.
    3.  Seed an imagination trajectory at t=0 with the real posterior.
        Roll forward T-1 steps calling `obs_step` with the real actions
        and real next-step embeddings, but continue the state with the
        imagined PRIOR (not the real posterior) so the trajectory drifts.
        At each rolled step we record
            c_{k+1} = KL( post_{k+1}  ||  sg(prior_{k+1}) )       (B,)
            feat_{k+1} = rssm.get_feat(prior_{k+1})                 (B, F)
        This is the exact one-step surprise of a drifted imagination —
        the same quantity the online Guardian will face when the VLA
        produces an action.
    4.  Compute per-step discounted returns
            C_target[k] = Σ_{j=k}^{K-1} γ^(j-k) · c_{j+1}
    5.  Critic regresses softplus( MLP(feat_{k+1}) ) toward C_target[k]
        with MSE.  Encoder / RSSM / decoder stay frozen throughout.

At inference the Guardian's operation is the mirror image:
    state ← real posterior of last real observation
    imag ← rssm.img_step(state, VLA_action)
    score ← critic(rssm.get_feat(imag))
    block if score ≥ threshold.
Training and inference therefore see the same distribution over feats.
"""

from __future__ import annotations
import argparse
import pathlib
import sys
import time
from typing import Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from models.safety_critic import SafetyCritic
from data.poison_dataset import MixedLiberoDataset


# ── Config helpers ─────────────────────────────────────────────────────────

def load_config(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        keys = k.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
        node[keys[-1]] = v
    return cfg


# ── Frozen world-model loader ──────────────────────────────────────────────

def load_frozen_world_model(
    rssm_cfg: dict, rssm_ckpt: str, device: str
) -> Tuple[RobotObsEncoder, RSSM, int]:
    encoder = RobotObsEncoder.from_config(rssm_cfg).to(device)
    rssm = RSSM(
        stoch=rssm_cfg["rssm"]["stoch"],
        deter=rssm_cfg["rssm"]["deter"],
        hidden=rssm_cfg["rssm"]["hidden"],
        rec_depth=rssm_cfg["rssm"]["rec_depth"],
        discrete=rssm_cfg["rssm"]["discrete"],
        act=rssm_cfg["rssm"]["act"],
        norm=rssm_cfg["rssm"]["norm"],
        mean_act=rssm_cfg["rssm"]["mean_act"],
        std_act=rssm_cfg["rssm"]["std_act"],
        min_std=rssm_cfg["rssm"]["min_std"],
        unimix_ratio=rssm_cfg["rssm"]["unimix_ratio"],
        initial=rssm_cfg["rssm"]["initial"],
        num_actions=rssm_cfg["rssm"]["num_actions"],
        embed=encoder.embed_dim,
        device=device,
    ).to(device)

    ckpt = torch.load(rssm_ckpt, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])

    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in rssm.parameters():
        p.requires_grad_(False)
    encoder.eval()
    rssm.eval()

    feat_size = rssm._stoch * max(rssm._discrete, 1) + rssm._deter
    return encoder, rssm, feat_size


# ── Imagination rollout + cost signal ──────────────────────────────────────

def _step_kl(post: dict, prior: dict, rssm: RSSM) -> torch.Tensor:
    """
    KL( post || sg(prior) ) summed across non-batch dims.
    This is the per-step world-model surprise (a.k.a. rep_loss in Dreamer's
    kl_loss, without the free-bits clip).

    Returns: (B,)
    """
    sg_prior = {k: v.detach() for k, v in prior.items()}
    dist_p = rssm.get_dist(post)
    dist_q = rssm.get_dist(sg_prior)
    if rssm._discrete:
        kl = torch.distributions.kl.kl_divergence(dist_p, dist_q)   # (B, stoch)
    else:
        kl = torch.distributions.kl.kl_divergence(dist_p._dist, dist_q._dist)  # (B,)
    while kl.dim() > 1:
        kl = kl.sum(dim=-1)
    return kl


@torch.no_grad()
def imagine_and_cost(
    encoder: RobotObsEncoder,
    rssm: RSSM,
    batch: dict,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce per-step imagined feats and per-step KL surprise along the
    drifted imagination trajectory.

    Returns:
        feats  (B, K, feat_size)   — imagined-prior feats for k = 1..T-1
        costs  (B, K)              — one-step KL surprise at each of those steps
    """
    obs = {
        "image": batch["image"].to(device),
        "proprio": batch["proprio"].to(device),
    }
    action = batch["action"].to(device)
    is_first = batch["is_first"].to(device)
    B, T = action.shape[0], action.shape[1]

    embed = encoder(obs)                              # (B, T, E)

    # Real posteriors — used ONLY to seed the imagination at t=0.
    post_real, _ = rssm.observe(embed, action, is_first)

    # Seed from t=0 real posterior.
    state = {k: v[:, 0] for k, v in post_real.items()}

    zero_is_first = torch.zeros(B, device=device)     # (B,)
    feats_list: list[torch.Tensor] = []
    costs_list: list[torch.Tensor] = []

    for k in range(T - 1):
        # obs_step(prev_state, action_k, embed_{k+1}, is_first) internally
        # computes prior_{k+1} = img_step(prev_state, action_k) and
        # post_{k+1} = posterior conditioned on (prior_{k+1}.deter, embed_{k+1}).
        post_k1, prior_k1 = rssm.obs_step(
            state,
            action[:, k],
            embed[:, k + 1],
            zero_is_first,
            sample=False,
        )
        costs_list.append(_step_kl(post_k1, prior_k1, rssm))       # (B,)
        feats_list.append(rssm.get_feat(prior_k1))                 # (B, F)
        # Continue the rollout on the PRIOR — this keeps the trajectory
        # in pure imagination so the world-model error compounds.
        state = prior_k1

    feats = torch.stack(feats_list, dim=1)    # (B, T-1, F)
    costs = torch.stack(costs_list, dim=1)    # (B, T-1)
    return feats, costs


def discounted_return(costs: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    C_target[k] = Σ_{j=k}^{K-1} γ^(j-k) · costs[j]

    Simple truncated Monte-Carlo (no value bootstrap).  costs is (B, K).
    """
    B, K = costs.shape
    out = torch.zeros_like(costs)
    running = torch.zeros(B, device=costs.device, dtype=costs.dtype)
    for k in reversed(range(K)):
        running = costs[:, k] + gamma * running
        out[:, k] = running
    return out


# ── Training loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/critic.yaml")
    parser.add_argument("--rssm_config", default="configs/rssm.yaml")
    parser.add_argument("override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    rssm_cfg = load_config(args.rssm_config, [])

    t_cfg = cfg["training"]
    device = t_cfg["device"]
    torch.manual_seed(t_cfg.get("seed", 0))

    # 1. Frozen world model
    encoder, rssm, feat_size = load_frozen_world_model(
        rssm_cfg, cfg["paths"]["rssm_checkpoint"], device
    )
    print(f"[train_critic] feat_size={feat_size}, frozen encoder+rssm loaded")

    # 2. Critic — the only trainable module
    critic = SafetyCritic.from_config(cfg, feat_size=feat_size).to(device)
    opt = torch.optim.AdamW(
        critic.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg.get("weight_decay", 0.0),
    )

    # 3. Mixed clean+poison dataset (labels unused)
    full = MixedLiberoDataset(
        clean_root=cfg["paths"]["clean_root"],
        poison_root=cfg["paths"]["poison_root"],
        window_len=t_cfg["window_len"],
        image_size=t_cfg["image_size"],
        seed=t_cfg.get("seed", 0),
    )
    val_len = max(1, int(len(full) * t_cfg.get("val_ratio", 0.05)))
    train_ds, val_ds = random_split(
        full, [len(full) - val_len, val_len],
        generator=torch.Generator().manual_seed(t_cfg.get("seed", 0)),
    )

    # Balanced sampler over the TRAIN split only.
    train_tags = [full.items[i][3] for i in train_ds.indices]
    import numpy as np
    tags = np.array(train_tags, dtype=np.int64)
    n_po = max(1, int((tags == 1).sum()))
    n_cl = max(1, int((tags == 0).sum()))
    weights = np.where(tags == 1, 0.5 / n_po, 0.5 / n_cl).astype(np.float32)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights).double(),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], sampler=sampler,
        num_workers=t_cfg.get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg.get("num_workers", 4), pin_memory=True, drop_last=True,
    )

    # 4. Logging
    logdir = pathlib.Path(cfg["paths"]["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))

    gamma = float(cfg["critic"]["gamma"])
    grad_clip = float(t_cfg.get("grad_clip", 100.0))
    total_steps = int(t_cfg["steps"])
    log_every = int(t_cfg["log_every"])
    val_every = int(t_cfg.get("val_every", 1000))
    save_every = int(t_cfg["save_every"])

    ckpt_path = pathlib.Path(cfg["paths"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 5. Training loop (step-based; loader is an infinite iterator)
    data_iter = iter(train_loader)
    t0 = time.time()
    print(f"[train_critic] training for {total_steps} steps on {device}")

    for step in range(total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # World-model rollout + cost — no grad.
        feats, costs = imagine_and_cost(encoder, rssm, batch, device)
        target = discounted_return(costs, gamma).detach()
        feats = feats.detach()

        # Critic update — grad flows only through the critic MLP.
        pred = critic(feats)                                      # (B, K)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        opt.step()

        if step % log_every == 0:
            source = batch["source"]
            with torch.no_grad():
                clean_mask = (source == 0)
                poison_mask = (source == 1)
                c_mean = costs.mean().item()
                c_clean = costs[clean_mask].mean().item() if clean_mask.any() else float("nan")
                c_poison = costs[poison_mask].mean().item() if poison_mask.any() else float("nan")
                p_mean = pred.mean().item()
                elapsed = time.time() - t0

            writer.add_scalar("critic/loss", loss.item(), step)
            writer.add_scalar("critic/cost_mean", c_mean, step)
            writer.add_scalar("critic/cost_clean", c_clean, step)
            writer.add_scalar("critic/cost_poison", c_poison, step)
            writer.add_scalar("critic/pred_mean", p_mean, step)
            print(
                f"[step {step:>6d}] loss={loss.item():.4f} "
                f"cost(clean/poison)={c_clean:.3f}/{c_poison:.3f} "
                f"pred={p_mean:.3f} elapsed={elapsed:.0f}s"
            )

        if step > 0 and step % val_every == 0:
            critic.eval()
            with torch.no_grad():
                vl_losses = []
                for vb in val_loader:
                    f_v, c_v = imagine_and_cost(encoder, rssm, vb, device)
                    tgt_v = discounted_return(c_v, gamma)
                    vl_losses.append(F.mse_loss(critic(f_v), tgt_v).item())
                val_loss = sum(vl_losses) / max(len(vl_losses), 1)
                writer.add_scalar("critic/val_loss", val_loss, step)
                print(f"[step {step:>6d}] val_loss={val_loss:.4f}")
            critic.train()

        if step > 0 and step % save_every == 0:
            torch.save(
                {"step": step, "critic": critic.state_dict(), "feat_size": feat_size},
                ckpt_path,
            )
            print(f"[train_critic] saved → {ckpt_path}")

    # Final save
    torch.save(
        {"step": total_steps, "critic": critic.state_dict(), "feat_size": feat_size},
        ckpt_path,
    )
    writer.close()
    print("[train_critic] done.")


if __name__ == "__main__":
    main()
