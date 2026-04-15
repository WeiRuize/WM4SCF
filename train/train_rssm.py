"""
train_rssm.py
=============
World-model training script for SafeVLA.

What it trains:
    RobotObsEncoder + RSSM + RobotObsDecoder jointly.
    Standard DreamerV3-style ELBO: image/proprio reconstruction + KL divergence.

Run on server:
    cd <project_root>
    python train/train_rssm.py --config configs/rssm.yaml [--override key=value ...]

Outputs:
    checkpoints/rssm.pt       — full world-model checkpoint (encoder + rssm + decoder)
    logs/rssm/                — TensorBoard logs
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# ── Path setup ─────────────────────────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.decoder import RobotObsDecoder
from models.rssm import RSSM
from models.dreamer_utils import Optimizer
from data.dataset import make_dataloader


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
        # Try to cast to numeric types
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


# ── Build components ───────────────────────────────────────────────────────

def build_world_model(cfg: dict, device: str):
    encoder = RobotObsEncoder.from_config(cfg)
    encoder = encoder.to(device)

    rssm_cfg = cfg["rssm"]
    rssm = RSSM(
        stoch=rssm_cfg["stoch"],
        deter=rssm_cfg["deter"],
        hidden=rssm_cfg["hidden"],
        rec_depth=rssm_cfg["rec_depth"],
        discrete=rssm_cfg["discrete"],
        act=rssm_cfg["act"],
        norm=rssm_cfg["norm"],
        mean_act=rssm_cfg["mean_act"],
        std_act=rssm_cfg["std_act"],
        min_std=rssm_cfg["min_std"],
        unimix_ratio=rssm_cfg["unimix_ratio"],
        initial=rssm_cfg["initial"],
        num_actions=rssm_cfg["num_actions"],
        embed=encoder.embed_dim,
        device=device,
    ).to(device)

    # Compute RSSM feature size
    feat_size = rssm_cfg["stoch"] * rssm_cfg["discrete"] + rssm_cfg["deter"]

    decoder = RobotObsDecoder.from_config(cfg, feat_size=feat_size)
    decoder = decoder.to(device)

    return encoder, rssm, decoder, feat_size


# ── Training step ──────────────────────────────────────────────────────────

def train_step(
    encoder, rssm, decoder,
    optimizer,
    batch: dict,
    cfg: dict,
    device: str,
    use_amp: bool,
) -> dict:
    t_cfg = cfg["training"]
    rssm_cfg = cfg["rssm"]

    obs = {
        "image": batch["image"].to(device),          # (B, T, H, W, C)
        "proprio": batch["proprio"].to(device),      # (B, T, proprio_dim)
    }
    action = batch["action"].to(device)               # (B, T, action_dim)
    is_first = batch["is_first"].to(device)           # (B, T)

    all_params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())

    with torch.cuda.amp.autocast(enabled=use_amp):
        # Encode
        embed = encoder(obs)                                      # (B, T, embed_dim)

        # World-model observe
        post, prior = rssm.observe(embed, action, is_first)
        feat = rssm.get_feat(post)                                # (B, T, feat_size)

        # Decode
        dists = decoder(feat)

        # Reconstruction losses
        image_loss = -dists["image"].log_prob(obs["image"]).mean()
        proprio_loss = -dists["proprio"].log_prob(obs["proprio"]).mean()
        recon_loss = image_loss + proprio_loss

        # KL divergence
        kl_loss, _, dyn_loss, rep_loss = rssm.kl_loss(
            post, prior,
            free=t_cfg["kl_free"],
            dyn_scale=t_cfg["dyn_scale"],
            rep_scale=t_cfg["rep_scale"],
        )
        kl_loss = kl_loss.mean()

        total_loss = recon_loss + kl_loss

    metrics = optimizer(total_loss, all_params)

    metrics.update({
        "image_loss": image_loss.item(),
        "proprio_loss": proprio_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "dyn_loss": dyn_loss.mean().item(),
        "rep_loss": rep_loss.mean().item(),
    })
    return metrics


# ── Checkpoint I/O ─────────────────────────────────────────────────────────

def save_checkpoint(path: str, encoder, rssm, decoder, step: int):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "encoder_state_dict": encoder.state_dict(),
        "rssm_state_dict": rssm.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
    }, path)
    print(f"[train_rssm] Saved checkpoint → {path} (step {step})")


def load_checkpoint(path: str, encoder, rssm, decoder, device: str) -> int:
    ckpt = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    rssm.load_state_dict(ckpt["rssm_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    step = ckpt.get("step", 0)
    print(f"[train_rssm] Loaded checkpoint from {path} (step {step})")
    return step


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rssm.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("override", nargs="*", help="key=value overrides, e.g. training.batch_size=8")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    t_cfg = cfg["training"]

    device = t_cfg["device"]
    use_amp = t_cfg["precision"] == 16
    torch.manual_seed(t_cfg.get("seed", 0))

    # Build
    encoder, rssm, decoder, feat_size = build_world_model(cfg, device)
    print(f"[train_rssm] Encoder embed_dim={encoder.embed_dim}, feat_size={feat_size}")

    # Optimizer (all parameters jointly)
    all_params = list(encoder.parameters()) + list(rssm.parameters()) + list(decoder.parameters())
    optimizer = Optimizer(
        "world_model", all_params,
        lr=t_cfg["model_lr"],
        eps=t_cfg["opt_eps"],
        clip=t_cfg["grad_clip"],
        wd=t_cfg.get("weight_decay", 0.0),
        opt=t_cfg.get("opt", "adam"),
        use_amp=use_amp,
    )

    # DataLoader
    loader = make_dataloader(
        cfg["paths"]["traindir"],
        window_len=t_cfg["batch_length"],
        batch_size=t_cfg["batch_size"],
        seed=t_cfg.get("seed", 0),
        labeled=False,
        num_workers=4,
    )
    data_iter = iter(loader)

    # Logging
    logdir = pathlib.Path(cfg["paths"]["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, encoder, rssm, decoder, device)

    total_steps = int(t_cfg["steps"])
    log_every = int(t_cfg["log_every"])
    save_every = int(t_cfg["save_every"])
    ckpt_path = cfg["paths"]["checkpoint"]

    print(f"[train_rssm] Training for {total_steps} steps on {device}")
    t0 = time.time()

    for step in range(start_step, total_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        metrics = train_step(encoder, rssm, decoder, optimizer, batch, cfg, device, use_amp)

        if step % log_every == 0:
            elapsed = time.time() - t0
            fps = (step - start_step + 1) * t_cfg["batch_size"] / max(elapsed, 1e-6)
            print(
                f"[step {step:>7d}]  total={metrics['total_loss']:.4f}  "
                f"img={metrics['image_loss']:.4f}  kl={metrics['kl_loss']:.4f}  "
                f"fps={fps:.1f}"
            )
            for k, v in metrics.items():
                writer.add_scalar(f"rssm/{k}", v, step)

        if step > 0 and step % save_every == 0:
            save_checkpoint(ckpt_path, encoder, rssm, decoder, step)

    save_checkpoint(ckpt_path, encoder, rssm, decoder, total_steps)
    writer.close()
    print("[train_rssm] Training complete.")


if __name__ == "__main__":
    main()
