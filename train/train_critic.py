"""
train_critic.py
===============
阶段二：Safety Critic training.

Pipeline (PROJECT_OVERVIEW §模块二):
    1. Load pre-encoded latent episodes from data/encode_latents.py output.
    2. Compute TD(λ) accumulated discounted costs for each latent step.
    3. Regress SafetyCritic(latent_t) → C_target_t with MSE.
    4. Update Lagrangian log_λ to enforce E[C] <= budget.

The RSSM/encoder checkpoint is NEVER loaded here; we operate purely on the
cached .pt files.  This makes critic training cheap and reproducible.
"""

from __future__ import annotations
import argparse
import pathlib
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.safety_critic import SafetyCritic, LagrangianSafetyCritic


class LatentEpisodeDataset(Dataset):
    """
    Each item is one pre-encoded episode:
        latent (T, feat), costs (T,), is_poison (scalar)
    We train step-wise, so __getitem__ returns all timesteps flattened; the
    collate_fn stacks them into a single (N, feat) batch per call.
    """

    def __init__(self, root: str | pathlib.Path):
        self.files = sorted(pathlib.Path(root).glob("*.pt"))
        if not self.files:
            raise RuntimeError(f"[LatentEpisodeDataset] No .pt under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.files[idx], map_location="cpu")


def td_lambda_cost(costs: torch.Tensor, horizon: int, gamma: float) -> torch.Tensor:
    """
    C_target_t = Σ_{k=0}^{K} γ^k · c_{t+k}      truncated discounted return.

    Args:
        costs:   (T,) per-step costs in [0, 1]
        horizon: truncation horizon K
        gamma:   discount factor
    Returns:
        targets: (T,)
    """
    T = costs.shape[0]
    out = torch.zeros_like(costs)
    for k in range(horizon + 1):
        idx = torch.arange(T)
        src = torch.clamp(idx + k, max=T - 1)
        out = out + (gamma ** k) * costs[src]
    return out


def flatten_batch(episodes: list[dict], horizon: int, gamma: float):
    latents, targets = [], []
    for ep in episodes:
        latent = ep["latent"].float()
        cost = ep["costs"].float()
        tgt = td_lambda_cost(cost, horizon, gamma)
        latents.append(latent)
        targets.append(tgt)
    return torch.cat(latents, dim=0), torch.cat(targets, dim=0)


def collate_latent(batch: list[dict]) -> list[dict]:
    return batch   # stash; flatten_batch handles it


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/critic.yaml")
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Minimal override loop; mirror train_rssm's behaviour.
    for ov in args.override:
        k, v = ov.split("=", 1)
        keys = k.split(".")
        node = cfg
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        node[keys[-1]] = v

    t_cfg = cfg["training"]
    device = t_cfg["device"]
    torch.manual_seed(t_cfg.get("seed", 0))

    # Feature size is written into the cached episodes.
    ds = LatentEpisodeDataset(args.latent_dir)
    feat_size = torch.load(ds.files[0])["latent"].shape[-1]
    print(f"[train_critic] feat_size={feat_size}  episodes={len(ds)}")

    val_len = max(1, int(len(ds) * t_cfg["val_split"]))
    train_ds, val_ds = random_split(
        ds, [len(ds) - val_len, val_len],
        generator=torch.Generator().manual_seed(t_cfg.get("seed", 0)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=2, collate_fn=collate_latent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=2, collate_fn=collate_latent,
    )

    critic = SafetyCritic.from_config(cfg, feat_size=feat_size).to(device)
    lag = LagrangianSafetyCritic(
        critic,
        budget=cfg["lagrangian"]["budget"],
        lambda_init=cfg["lagrangian"]["lambda_init"],
        lambda_max=cfg["lagrangian"]["lambda_max"],
    ).to(device)

    opt_critic = torch.optim.Adam(
        critic.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"],
    )
    opt_lambda = torch.optim.Adam([lag.log_lambda], lr=cfg["lagrangian"]["lambda_lr"])

    logdir = pathlib.Path(cfg["paths"]["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))

    horizon = cfg["critic"]["td_horizon"]
    gamma = cfg["critic"]["gamma"]
    epochs = t_cfg["epochs"]
    step = 0
    t0 = time.time()

    for epoch in range(epochs):
        critic.train()
        for episodes in train_loader:
            latent, target = flatten_batch(episodes, horizon, gamma)
            latent = latent.to(device)
            target = target.to(device)

            loss_c = lag.critic_loss(latent, target)
            opt_critic.zero_grad()
            loss_c.backward()
            opt_critic.step()

            loss_lag = lag.lagrangian_loss(target)
            opt_lambda.zero_grad()
            loss_lag.backward()
            opt_lambda.step()

            if step % t_cfg["log_every"] == 0:
                writer.add_scalar("critic/loss", loss_c.item(), step)
                writer.add_scalar("critic/lambda", lag.lambda_value.item(), step)
                writer.add_scalar("critic/cost_mean", target.mean().item(), step)
                print(f"[ep {epoch:>3d} step {step:>6d}] "
                      f"L_c={loss_c.item():.4f}  λ={lag.lambda_value.item():.3f}  "
                      f"cost_mean={target.mean().item():.3f}")
            step += 1

        # Validation
        critic.eval()
        with torch.no_grad():
            vl_losses = []
            for episodes in val_loader:
                latent, target = flatten_batch(episodes, horizon, gamma)
                latent = latent.to(device)
                target = target.to(device)
                vl_losses.append(lag.critic_loss(latent, target).item())
            val_loss = sum(vl_losses) / max(len(vl_losses), 1)
            writer.add_scalar("critic/val_loss", val_loss, step)
            print(f"[ep {epoch:>3d}] val_loss={val_loss:.4f} elapsed={time.time() - t0:.0f}s")

        if (epoch + 1) % t_cfg["save_every"] == 0 or epoch == epochs - 1:
            ckpt = {
                "epoch": epoch + 1,
                "critic": critic.state_dict(),
                "log_lambda": lag.log_lambda.detach().cpu(),
                "feat_size": feat_size,
            }
            out = pathlib.Path(cfg["paths"]["checkpoint"])
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out)
            print(f"[train_critic] Saved → {out}")

    writer.close()
    print("[train_critic] Training complete.")


if __name__ == "__main__":
    main()
