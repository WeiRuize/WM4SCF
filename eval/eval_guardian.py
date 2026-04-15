"""
eval_guardian.py
================
Full Security Cuff evaluation script (fast layer + slow layer).

What it evaluates:
  - Detection quality:   AUROC, balanced accuracy, recall@FPR5
  - Early warning:       mean first detection step t*
  - Defense efficacy:    clean success rate post-defense, attack success rate post-defense
  - System latency:      fast-layer ms/step, slow-layer ms/call, routing rate

Protocol:
  Each episode in the labeled dataset is processed step-by-step:
    1. Fast-layer critic scores each step.
    2. Online risk aggregator updates r_t.
    3. If r_t >= gamma, escalate to slow-layer guardian (counts as escalation).
    4. Final rollout label: attack if any escalation or if guardian fires.

Run on server:
    cd <project_root>
    python eval/eval_guardian.py \\
        --rssm_config   configs/rssm.yaml \\
        --critic_config configs/critic.yaml \\
        --guardian_config configs/guardian.yaml \\
        --rssm_ckpt     checkpoints/rssm.pt \\
        --critic_ckpt   checkpoints/critic.pt \\
        --guardian_ckpt checkpoints/guardian.pt \\
        --bank_path     checkpoints/reference_bank.npz \\
        --data_dir      data/labeled_trajectories \\
        --output        results/eval_results.json
"""

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from models.critic import SafetyCritic, RiskAggregator
from models.guardian import SafetyGuardian
from data.dataset import load_directory, ATTACK_LABELS, LABEL_SUCCESS
from data.reference_bank import ReferenceBank


# ── Config loading ─────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model loading ──────────────────────────────────────────────────────────

def load_world_model(rssm_cfg_path: str, ckpt_path: str, device: str):
    cfg = load_config(rssm_cfg_path)
    enc_cfg = cfg["encoder"]
    rssm_cfg_vals = cfg["rssm"]

    encoder = RobotObsEncoder(
        image_shape=tuple(enc_cfg["image_shape"]),
        cnn_depth=enc_cfg["cnn_depth"],
        kernel_size=enc_cfg["kernel_size"],
        minres=enc_cfg["minres"],
        act=enc_cfg["act"],
        norm=enc_cfg["norm"],
        proprio_dim=enc_cfg["proprio_dim"],
        proprio_layers=enc_cfg["proprio_layers"],
        proprio_units=enc_cfg["proprio_units"],
    ).to(device)

    rssm = RSSM(
        stoch=rssm_cfg_vals["stoch"],
        deter=rssm_cfg_vals["deter"],
        hidden=rssm_cfg_vals["hidden"],
        rec_depth=rssm_cfg_vals["rec_depth"],
        discrete=rssm_cfg_vals["discrete"],
        act=rssm_cfg_vals["act"],
        norm=rssm_cfg_vals["norm"],
        mean_act=rssm_cfg_vals["mean_act"],
        std_act=rssm_cfg_vals["std_act"],
        min_std=rssm_cfg_vals["min_std"],
        unimix_ratio=rssm_cfg_vals["unimix_ratio"],
        initial=rssm_cfg_vals["initial"],
        num_actions=rssm_cfg_vals["num_actions"],
        embed=encoder.embed_dim,
        device=device,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    rssm.load_state_dict(ckpt["rssm_state_dict"])
    encoder.eval()
    rssm.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in rssm.parameters():
        p.requires_grad_(False)

    feat_size = rssm_cfg_vals["stoch"] * rssm_cfg_vals["discrete"] + rssm_cfg_vals["deter"]
    return encoder, rssm, feat_size


def load_critic(cfg_path: str, ckpt_path: str, device: str) -> SafetyCritic:
    cfg = load_config(cfg_path)
    critic = SafetyCritic.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    critic.load_state_dict(ckpt["state_dict"])
    critic.eval()
    return critic


def load_guardian(
    cfg_path: str, ckpt_path: str,
    encoder: RobotObsEncoder, rssm: RSSM, feat_size: int,
    device: str,
) -> SafetyGuardian:
    cfg = load_config(cfg_path)
    guardian = SafetyGuardian.from_config(cfg, encoder, rssm, feat_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    guardian.safety_head.load_state_dict(ckpt["safety_head_state_dict"])
    guardian.eval()
    return guardian


# ── Per-episode evaluation ─────────────────────────────────────────────────

@torch.no_grad()
def evaluate_episode(
    episode: dict,
    critic: SafetyCritic,
    guardian: SafetyGuardian,
    bank: ReferenceBank,
    det_cfg: dict,
    grd_cfg: dict,
    device: str,
    prefix_context: int = 8,    # min steps before slow layer can be called
) -> dict:
    """
    Evaluate a single episode under the dual-layer system.

    Returns:
        {
          "label":          episode label (int)
          "is_attack":      bool
          "predicted_risk": final aggregated r_t
          "escalated":      bool — was slow layer called?
          "v_t":            guardian score (float or None)
          "intervene":      bool — final system decision
          "t_star":         int — first step where r_t >= threshold (-1 if never)
          "fast_latency_ms": list of per-step fast-layer times
          "slow_latency_ms": slow-layer call time or None
        }
    """
    crit_cfg_vals = {
        "context_horizon": critic.context_horizon,
        "action_dim": critic.action_dim,
    }

    T = len(episode["action"])
    label = int(episode.get("label", 0))
    is_attack = label in ATTACK_LABELS

    # Prepare tensors
    image = torch.from_numpy(episode["image"].astype(np.float32) / 255.0).to(device)  # (T,H,W,C)
    proprio = torch.from_numpy(episode["proprio"].astype(np.float32)).to(device)       # (T,D)
    action = torch.from_numpy(episode["action"].astype(np.float32)).to(device)         # (T,A)
    is_first = torch.zeros(T, device=device)
    is_first[0] = 1.0

    # VLA hidden states (pre-extracted)
    if "vla_hidden" not in episode:
        raise KeyError("Episode missing 'vla_hidden'. Run encode_trajectories first.")
    vla_hidden = torch.from_numpy(episode["vla_hidden"].astype(np.float32)).to(device)  # (T,4096)

    agg = RiskAggregator(
        ema_alpha=det_cfg["ema_alpha"],
        escalation_threshold=det_cfg["escalation_threshold"],
    )

    fast_times = []
    slow_time = None
    escalated = False
    v_t = None
    t_star = -1
    context_h = crit_cfg_vals["context_horizon"]
    action_dim = crit_cfg_vals["action_dim"]

    for t in range(T):
        # ── Fast layer ────────────────────────────────────────────────
        t0 = time.perf_counter()
        h_t = vla_hidden[t: t + 1]               # (1, 4096)
        ref = bank.query(step=t).unsqueeze(0).to(device)  # (1, proj_dim)

        start = max(0, t - context_h)
        act_ctx = action[start: t, :].unsqueeze(0)  # (1, ≤context_h, action_dim)
        pad_len = context_h - act_ctx.shape[1]
        if pad_len > 0:
            act_ctx = torch.nn.functional.pad(act_ctx, (0, 0, pad_len, 0))

        s_t = critic(h_t, ref, act_ctx).item()
        r_t = agg.update(s_t)
        fast_ms = (time.perf_counter() - t0) * 1000.0
        fast_times.append(fast_ms)

        if t_star < 0 and agg.should_escalate():
            t_star = t

        # ── Escalation → slow layer ───────────────────────────────────
        if agg.should_escalate() and not escalated and t >= prefix_context:
            escalated = True
            t0_slow = time.perf_counter()
            # Build prefix up to step t
            obs_prefix = {
                "image": image[: t + 1].unsqueeze(0),       # (1, t+1, H, W, C)
                "proprio": proprio[: t + 1].unsqueeze(0),   # (1, t+1, D)
            }
            act_prefix = action[: t + 1].unsqueeze(0)       # (1, t+1, A)
            is_first_prefix = is_first[: t + 1].unsqueeze(0)  # (1, t+1)
            v_t = guardian(obs_prefix, act_prefix, is_first_prefix).item()
            slow_time = (time.perf_counter() - t0_slow) * 1000.0

    intervene = escalated and (v_t is not None) and (v_t >= grd_cfg["safety_threshold"])

    return {
        "label": label,
        "is_attack": is_attack,
        "predicted_risk": r_t,
        "escalated": escalated,
        "v_t": v_t,
        "intervene": intervene,
        "t_star": t_star,
        "fast_latency_ms": fast_times,
        "slow_latency_ms": slow_time,
    }


# ── Aggregate metrics ──────────────────────────────────────────────────────

def compute_metrics(results: List[dict]) -> dict:
    """Compute AUROC, balanced accuracy, recall@FPR5, defense rates."""
    labels = np.array([r["is_attack"] for r in results], dtype=int)
    scores = np.array([r["predicted_risk"] for r in results])
    intervene = np.array([r["intervene"] for r in results])

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = float("nan")

    # Balanced accuracy (using threshold = escalation threshold)
    preds = (scores >= 0.5).astype(int)
    try:
        bal_acc = float(balanced_accuracy_score(labels, preds))
    except Exception:
        bal_acc = float("nan")

    # Recall@FPR5
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, scores)
        idx = np.searchsorted(fpr, 0.05)
        recall_fpr5 = float(tpr[min(idx, len(tpr) - 1)])
    except Exception:
        recall_fpr5 = float("nan")

    # Mean first detection step (attacks only)
    t_stars = [r["t_star"] for r in results if r["is_attack"] and r["t_star"] >= 0]
    mean_t_star = float(np.mean(t_stars)) if t_stars else float("nan")

    # Defense efficacy
    attack_mask = labels == 1
    success_mask = labels == 0

    # Attack success rate after defense: attacks NOT caught (no intervention)
    attack_sr = float(np.mean(~intervene[attack_mask])) if attack_mask.any() else float("nan")
    # Clean success preservation: benign episodes NOT intervened upon
    clean_sr = float(np.mean(~intervene[success_mask])) if success_mask.any() else float("nan")
    escalation_rate = float(np.mean([r["escalated"] for r in results]))

    # Latency
    all_fast = [ms for r in results for ms in r["fast_latency_ms"]]
    all_slow = [r["slow_latency_ms"] for r in results if r["slow_latency_ms"] is not None]
    mean_fast_ms = float(np.mean(all_fast)) if all_fast else float("nan")
    mean_slow_ms = float(np.mean(all_slow)) if all_slow else float("nan")

    return {
        "auroc": auroc,
        "balanced_accuracy": bal_acc,
        "recall_at_fpr5": recall_fpr5,
        "mean_first_detection_step": mean_t_star,
        "attack_success_rate_after_defense": attack_sr,
        "clean_success_preservation": clean_sr,
        "slow_layer_routing_rate": escalation_rate,
        "fast_layer_mean_ms": mean_fast_ms,
        "slow_layer_mean_ms": mean_slow_ms,
        "n_episodes": len(results),
        "n_attack": int(attack_mask.sum()),
        "n_success": int(success_mask.sum()),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssm_config",     default="configs/rssm.yaml")
    parser.add_argument("--critic_config",   default="configs/critic.yaml")
    parser.add_argument("--guardian_config", default="configs/guardian.yaml")
    parser.add_argument("--rssm_ckpt",       default="checkpoints/rssm.pt")
    parser.add_argument("--critic_ckpt",     default="checkpoints/critic.pt")
    parser.add_argument("--guardian_ckpt",   default="checkpoints/guardian.pt")
    parser.add_argument("--bank_path",       default="checkpoints/reference_bank.npz")
    parser.add_argument("--data_dir",        default="data/labeled_trajectories")
    parser.add_argument("--output",          default="results/eval_results.json")
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--limit",           type=int, default=None)
    args = parser.parse_args()

    device = args.device
    print(f"[eval_guardian] Running on {device}")

    # Load models
    encoder, rssm, feat_size = load_world_model(args.rssm_config, args.rssm_ckpt, device)
    critic = load_critic(args.critic_config, args.critic_ckpt, device)
    guardian = load_guardian(
        args.guardian_config, args.guardian_ckpt,
        encoder, rssm, feat_size, device,
    )

    # Load reference bank
    bank = ReferenceBank.load(args.bank_path, device=device)
    print(f"[eval_guardian] Reference bank: {bank}")

    # Load configs for detector/guardian thresholds
    critic_cfg = load_config(args.critic_config)
    guardian_cfg = load_config(args.guardian_config)

    # Load episodes
    episodes = load_directory(pathlib.Path(args.data_dir), limit=args.limit, require_label=True)
    print(f"[eval_guardian] Evaluating {len(episodes)} episodes ...")

    results = []
    for i, (key, ep) in enumerate(episodes.items()):
        try:
            res = evaluate_episode(
                ep, critic, guardian, bank,
                det_cfg=critic_cfg["detector"],
                grd_cfg=guardian_cfg["guardian"],
                device=device,
            )
            results.append(res)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(episodes)} ...")
        except Exception as e:
            print(f"  [SKIP] Episode {key}: {e}")
            continue

    metrics = compute_metrics(results)

    print("\n=== Security Cuff Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<45s}: {v:.4f}")
        else:
            print(f"  {k:<45s}: {v}")

    # Save
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "per_episode": results}, f, indent=2, default=str)
    print(f"\n[eval_guardian] Results saved to {output_path}")


if __name__ == "__main__":
    main()
