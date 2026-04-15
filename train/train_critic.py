"""
train_critic.py
===============
Safety Critic (fast layer) training script.

Pipeline:
  1. Load labeled trajectory dataset (data/labeled_trajectories/).
  2. Load frozen RSSM world model (for feature context, not directly used in critic).
  3. Load pretrained OpenVLA to extract hidden states h_t for each step.
     (Note: hidden states must be pre-extracted and stored; see scripts/encode_trajectories.sh)
  4. Build reference bank from success-labeled hidden states.
  5. Train SafetyCritic binary classifier (is_attack vs. success).

Pre-extraction assumption:
  Before calling this script, run scripts/encode_trajectories.sh to extract
  VLA hidden states for all episodes and store them alongside the trajectory
  data as .npz files with an extra key "vla_hidden" of shape (T, 4096).
  This avoids loading the full VLA during critic training.

Run on server:
    cd <project_root>
    python train/train_critic.py --config configs/critic.yaml [--override key=value ...]

Outputs:
    checkpoints/critic.pt       — SafetyCritic weights + projector
    checkpoints/reference_bank.npz
    logs/critic/                — TensorBoard logs
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

from models.critic import SafetyCritic, RiskAggregator
from data.dataset import LabeledTrajectoryDataset, collate_episodes, LABEL_SUCCESS
from data.reference_bank import ReferenceBank


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


# ── Dataset with pre-extracted VLA hidden states ──────────────────────────

class HiddenStateDataset(LabeledTrajectoryDataset):
    """
    Extends LabeledTrajectoryDataset to also load pre-extracted VLA hidden
    states (key "vla_hidden", shape (T, 4096)) from the .npz episode files.
    """

    def __getitem__(self, idx: int) -> dict:
        # super().__getitem__() calls sample_window and stores the window start
        # in self._last_window_start — reuse it to align the hidden state slice.
        out = super().__getitem__(idx)
        ep_key = self._keys[idx % len(self._keys)]
        ep = self.episodes[ep_key]
        if "vla_hidden" not in ep:
            raise KeyError(
                f"Episode {ep_key} is missing 'vla_hidden'. "
                "Run train/encode_trajectories.py first."
            )
        idx_start = self._last_window_start   # set by parent's sample_window call
        hidden_window = ep["vla_hidden"][idx_start: idx_start + self.window_len]
        if len(hidden_window) < self.window_len:
            pad = np.zeros(
                (self.window_len - len(hidden_window), ep["vla_hidden"].shape[1]),
                dtype=np.float32,
            )
            hidden_window = np.concatenate([hidden_window, pad], axis=0)
        out["vla_hidden"] = torch.from_numpy(hidden_window.astype(np.float32))
        return out


# ── Reference bank construction ────────────────────────────────────────────

@torch.no_grad()
def build_reference_bank(
    critic: SafetyCritic,
    dataset: HiddenStateDataset,
    cfg: dict,
    device: str,
) -> ReferenceBank:
    """Populate the reference bank with projected hidden states from success episodes."""
    bank_cfg = cfg["reference_bank"]
    crit_cfg = cfg["critic"]
    bank = ReferenceBank(
        capacity=bank_cfg["capacity"],
        proj_dim=crit_cfg["proj_dim"],
        device=device,
    )
    critic.eval()
    added = 0
    for key, ep in dataset.episodes.items():
        label = int(ep.get("label", -1))
        if label != LABEL_SUCCESS:
            continue
        if "vla_hidden" not in ep:
            continue
        h = torch.from_numpy(ep["vla_hidden"].astype(np.float32)).to(device)  # (T, 4096)
        h_proj = critic.project(h)                                              # (T, proj_dim)
        bank.add_episode(h_proj)
        added += 1
    print(f"[train_critic] Reference bank built from {added} success episodes, {len(bank)} vectors.")
    return bank


# ── Training loop ──────────────────────────────────────────────────────────

def train_epoch(
    critic: SafetyCritic,
    bank: ReferenceBank,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    device: str,
    pos_weight: torch.Tensor,
) -> dict:
    critic.train()
    crit_cfg = cfg["critic"]
    total_loss, total_correct, total_n = 0.0, 0, 0

    for batch in loader:
        vla_hidden = batch["vla_hidden"].to(device)     # (B, T, 4096)
        actions = batch["action"].to(device)            # (B, T, action_dim)
        labels = batch["is_attack"].to(device)          # (B,)  0 or 1
        B, T = vla_hidden.shape[:2]

        context_h = crit_cfg["context_horizon"]
        action_dim = crit_cfg["action_dim"]

        # Use middle step of window for per-window prediction
        # (could also predict per-step and pool; this is the simple variant)
        t_mid = T // 2
        h_t = vla_hidden[:, t_mid, :]                   # (B, 4096)

        # Reference: use global bank mean (could be made step-aligned)
        ref = bank.global_mean().unsqueeze(0).expand(B, -1).to(device)   # (B, proj_dim)

        # Action context: last context_h steps ending at t_mid
        start = max(0, t_mid - context_h)
        act_ctx = actions[:, start: t_mid, :]           # (B, ≤context_h, action_dim)
        # Pad if short
        pad_len = context_h - act_ctx.shape[1]
        if pad_len > 0:
            act_ctx = F.pad(act_ctx, (0, 0, pad_len, 0))

        scores = critic(h_t, ref, act_ctx)              # (B,)
        loss = F.binary_cross_entropy_with_logits(
            torch.logit(scores.clamp(1e-6, 1 - 1e-6)),
            labels.float(),
            pos_weight=pos_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optimizer.step()

        preds = (scores >= 0.5).long()
        correct = (preds == labels.long()).sum().item()
        total_loss += loss.item() * B
        total_correct += correct
        total_n += B

    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }


@torch.no_grad()
def validate(
    critic: SafetyCritic,
    bank: ReferenceBank,
    loader: DataLoader,
    cfg: dict,
    device: str,
) -> dict:
    critic.eval()
    crit_cfg = cfg["critic"]
    all_scores, all_labels = [], []

    for batch in loader:
        vla_hidden = batch["vla_hidden"].to(device)
        actions = batch["action"].to(device)
        labels = batch["is_attack"]
        B, T = vla_hidden.shape[:2]
        context_h = crit_cfg["context_horizon"]
        t_mid = T // 2
        h_t = vla_hidden[:, t_mid, :]
        ref = bank.global_mean().unsqueeze(0).expand(B, -1).to(device)
        start = max(0, t_mid - context_h)
        act_ctx = actions[:, start: t_mid, :]
        pad_len = context_h - act_ctx.shape[1]
        if pad_len > 0:
            act_ctx = F.pad(act_ctx, (0, 0, pad_len, 0))
        scores = critic(h_t, ref, act_ctx)
        all_scores.append(scores.cpu())
        all_labels.append(labels)

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    preds = (scores >= 0.5).long()
    acc = (preds == labels.long()).float().mean().item()

    # AUROC via simple threshold sweep
    auroc = _auroc(scores.numpy(), labels.numpy())
    return {"val_acc": acc, "val_auroc": auroc}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC via threshold sweep."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(path: str, critic: SafetyCritic, epoch: int):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": critic.state_dict()}, path)
    print(f"[train_critic] Saved checkpoint → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/critic.yaml")
    parser.add_argument("override", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    t_cfg = cfg["training"]
    device = t_cfg["device"]
    torch.manual_seed(t_cfg.get("seed", 0))

    # Build critic
    critic = SafetyCritic.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"[train_critic] Critic parameters: {n_params:,}")

    # Dataset
    full_ds = HiddenStateDataset(
        cfg["paths"]["data_dir"],
        window_len=64,
        seed=t_cfg.get("seed", 0),
    )
    val_size = max(1, int(len(full_ds) * t_cfg["val_split"]))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
                               collate_fn=collate_episodes, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
                             collate_fn=collate_episodes, num_workers=4)

    # Reference bank (from success episodes in full dataset)
    bank = build_reference_bank(critic, full_ds, cfg, device)
    bank.save(str(pathlib.Path(cfg["paths"]["checkpoint"]).parent / "reference_bank.npz"))

    # Optimizer
    optimizer = torch.optim.AdamW(critic.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    pos_weight = torch.tensor([t_cfg["pos_weight"]], device=device)

    # Logging
    logdir = pathlib.Path(cfg["paths"]["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    print(f"[train_critic] Training for {t_cfg['epochs']} epochs on {device}")
    best_auroc = 0.0

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_metrics = train_epoch(critic, bank, train_loader, optimizer, cfg, device, pos_weight)
        val_metrics = validate(critic, bank, val_loader, cfg, device)

        print(
            f"[epoch {epoch:>3d}/{t_cfg['epochs']}]  "
            f"loss={train_metrics['loss']:.4f}  acc={train_metrics['acc']:.3f}  "
            f"val_acc={val_metrics['val_acc']:.3f}  val_auroc={val_metrics['val_auroc']:.3f}"
        )

        for k, v in {**train_metrics, **val_metrics}.items():
            if not (isinstance(v, float) and v != v):   # skip NaN
                writer.add_scalar(f"critic/{k}", v, epoch)

        if val_metrics["val_auroc"] > best_auroc:
            best_auroc = val_metrics["val_auroc"]
            save_checkpoint(cfg["paths"]["checkpoint"], critic, epoch)

        if epoch % t_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                str(pathlib.Path(cfg["paths"]["checkpoint"]).with_stem(f"critic_ep{epoch}")),
                critic, epoch,
            )

    writer.close()
    print(f"[train_critic] Done. Best val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
