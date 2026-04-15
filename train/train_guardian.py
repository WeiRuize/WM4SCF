"""
train_guardian.py
=================
Safety Guardian (slow layer) training script.

Trains ONLY the SafetyHead MLP.
The RobotObsEncoder and RSSM remain fully frozen throughout.

Pipeline:
  1. Load frozen encoder + RSSM from RSSM checkpoint.
  2. For each labeled episode in the dataset:
       a. Encode observations → embed.
       b. rssm.observe() → posterior state sequence.
       c. Take last state as starting point.
       d. rssm.rollout_future() → imagined states.
       e. safety_head(get_feat(imagined)) → danger scores → v_t.
       f. Binary cross-entropy against episode label.

Run on server:
    cd <project_root>
    python train/train_guardian.py --config configs/guardian.yaml [--override key=value ...]

Outputs:
    checkpoints/guardian.pt   — SafetyHead weights only
    logs/guardian/            — TensorBoard logs
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from models.guardian import SafetyGuardian
from data.dataset import LabeledTrajectoryDataset, collate_episodes, ATTACK_LABELS


# ── Config helpers ────────────────────────────────────────────────────────

def load_config(path: str, overrides: list) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
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


# ── Model loading ──────────────────────────────────────────────────────────

def load_frozen_world_model(rssm_cfg: dict, ckpt_path: str, device: str):
    enc_cfg = rssm_cfg["encoder"]
    r_cfg = rssm_cfg["rssm"]

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
        stoch=r_cfg["stoch"],
        deter=r_cfg["deter"],
        hidden=r_cfg["hidden"],
        rec_depth=r_cfg["rec_depth"],
        discrete=r_cfg["discrete"],
        act=r_cfg["act"],
        norm=r_cfg["norm"],
        mean_act=r_cfg["mean_act"],
        std_act=r_cfg["std_act"],
        min_std=r_cfg["min_std"],
        unimix_ratio=r_cfg["unimix_ratio"],
        initial=r_cfg["initial"],
        num_actions=r_cfg["num_actions"],
        embed=encoder.embed_dim,
        device=device,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    rssm.load_state_dict(ckpt["rssm_state_dict"])
    encoder.eval()
    rssm.eval()

    feat_size = r_cfg["stoch"] * r_cfg["discrete"] + r_cfg["deter"]
    return encoder, rssm, feat_size


# ── Training loop ──────────────────────────────────────────────────────────

def run_epoch(
    guardian: SafetyGuardian,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    train: bool,
    pos_weight: torch.Tensor,
) -> dict:
    guardian.safety_head.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            obs = {
                "image": batch["image"].to(device),
                "proprio": batch["proprio"].to(device),
            }
            action = batch["action"].to(device)
            is_first = batch["is_first"].to(device)
            labels = batch["is_attack"].to(device)      # (B,)

            v_t = guardian(obs, action, is_first)       # (B,)

            loss = F.binary_cross_entropy_with_logits(
                torch.logit(v_t.clamp(1e-6, 1 - 1e-6)),
                labels,
                pos_weight=pos_weight,
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    guardian.safety_head.parameters(), 5.0
                )
                optimizer.step()

            B = labels.shape[0]
            preds = (v_t >= guardian.safety_threshold).long()
            correct = (preds == labels.long()).sum().item()
            total_loss += loss.item() * B
            total_correct += correct
            total_n += B

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels.numpy(), scores.numpy()))
    except Exception:
        return float("nan")


@torch.no_grad()
def validate_auroc(
    guardian: SafetyGuardian,
    loader: DataLoader,
    device: str,
) -> float:
    guardian.eval()
    all_v, all_y = [], []
    for batch in loader:
        obs = {"image": batch["image"].to(device), "proprio": batch["proprio"].to(device)}
        action = batch["action"].to(device)
        is_first = batch["is_first"].to(device)
        v_t = guardian(obs, action, is_first).cpu()
        all_v.append(v_t)
        all_y.append(batch["is_attack"])
    return _auroc(torch.cat(all_v), torch.cat(all_y))


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(path: str, guardian: SafetyGuardian, epoch: int):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "safety_head_state_dict": guardian.safety_head.state_dict(),
    }, path)
    print(f"[train_guardian] Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/guardian.yaml")
    parser.add_argument("--rssm_config", default="configs/rssm.yaml")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    g_cfg = load_config(args.config, args.override)
    rssm_cfg = load_config(args.rssm_config, [])
    t_cfg = g_cfg["training"]
    device = t_cfg["device"]
    torch.manual_seed(t_cfg.get("seed", 0))

    # Load frozen world model
    encoder, rssm, feat_size = load_frozen_world_model(
        rssm_cfg, g_cfg["paths"]["rssm_checkpoint"], device
    )
    print(f"[train_guardian] Loaded RSSM. feat_size={feat_size}")

    # Build guardian (only safety_head will be trained)
    g_vals = g_cfg["guardian"]
    guardian = SafetyGuardian(
        encoder=encoder,
        rssm=rssm,
        rollout_horizon=g_vals["rollout_horizon"],
        feat_size=feat_size,
        safety_head_layers=g_vals["safety_head_layers"],
        safety_head_units=g_vals["safety_head_units"],
        safety_head_act=g_vals["safety_head_act"],
        score_aggregation=g_vals["score_aggregation"],
        safety_threshold=g_vals["safety_threshold"],
    ).to(device)

    n_params = sum(p.numel() for p in guardian.safety_head.parameters())
    print(f"[train_guardian] Safety-head trainable params: {n_params:,}")

    # Dataset
    full_ds = LabeledTrajectoryDataset(
        g_cfg["paths"]["data_dir"], window_len=64, seed=t_cfg.get("seed", 0)
    )
    val_size = max(1, int(len(full_ds) * 0.1))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                               collate_fn=collate_episodes, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
                             collate_fn=collate_episodes, num_workers=4)

    # Optimizer — only safety_head params
    optimizer = torch.optim.AdamW(
        guardian.safety_head.parameters(),
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    pos_weight = torch.tensor([5.0], device=device)

    logdir = pathlib.Path(g_cfg["paths"]["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    best_auroc = 0.0
    for epoch in range(1, t_cfg["epochs"] + 1):
        train_m = run_epoch(guardian, train_loader, optimizer, device, train=True,
                            pos_weight=pos_weight)
        val_m = run_epoch(guardian, val_loader, None, device, train=False,
                          pos_weight=pos_weight)
        val_auroc = validate_auroc(guardian, val_loader, device)

        print(
            f"[epoch {epoch:>3d}/{t_cfg['epochs']}]  "
            f"loss={train_m['loss']:.4f}  acc={train_m['acc']:.3f}  "
            f"val_acc={val_m['acc']:.3f}  val_auroc={val_auroc:.3f}"
        )

        writer.add_scalar("guardian/train_loss", train_m["loss"], epoch)
        writer.add_scalar("guardian/train_acc", train_m["acc"], epoch)
        writer.add_scalar("guardian/val_acc", val_m["acc"], epoch)
        writer.add_scalar("guardian/val_auroc", val_auroc, epoch)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            save_checkpoint(g_cfg["paths"]["checkpoint"], guardian, epoch)

        if epoch % t_cfg.get("save_every", 5) == 0:
            p = str(pathlib.Path(g_cfg["paths"]["checkpoint"]).with_stem(f"guardian_ep{epoch}"))
            save_checkpoint(p, guardian, epoch)

    writer.close()
    print(f"[train_guardian] Done. Best val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
