"""
encode_latents.py
=================
Pre-encode every trajectory in the clean + poisoned splits into the frozen
RSSM latent space, and cache the result on disk for fast Safety Critic
training.

Output format (.pt per episode):
    latent:    (T, feat_size)  float32   — cat(stoch_flat, deter)
    costs:     (T,)             float32   — per-step unsafe label
    is_poison: scalar           int

The RSSM, encoder, and decoder are all loaded FROZEN from the阶段一
checkpoint.  This script never touches model weights.
"""

from __future__ import annotations
import argparse
import pathlib
import sys

import torch
import yaml
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from models.encoder import RobotObsEncoder
from models.rssm import RSSM
from data.poison_dataset import PoisonDataset


def build_model(cfg: dict, device: str):
    encoder = RobotObsEncoder.from_config(cfg).to(device).eval()
    rssm = RSSM(
        stoch=cfg["rssm"]["stoch"],
        deter=cfg["rssm"]["deter"],
        hidden=cfg["rssm"]["hidden"],
        discrete=cfg["rssm"]["discrete"],
        num_actions=cfg["rssm"]["num_actions"],
        embed=encoder.embed_dim,
        device=device,
    ).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in rssm.parameters():
        p.requires_grad_(False)
    return encoder, rssm


@torch.no_grad()
def encode_episode(episode: dict, encoder: RobotObsEncoder, rssm: RSSM, device: str) -> dict:
    image = episode["image"].unsqueeze(0).to(device)
    proprio = episode["proprio"].unsqueeze(0).to(device)
    action = episode["action"].unsqueeze(0).to(device)
    is_first = episode["is_first"].unsqueeze(0).to(device)

    embed = encoder({"image": image, "proprio": proprio})
    post, _ = rssm.observe(embed, action, is_first)
    feat = rssm.get_feat(post).squeeze(0).cpu()     # (T, feat_size)
    return {
        "latent": feat,
        "costs": episode["costs"].clone(),
        "is_poison": int(episode["is_poison"].item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--poison_root", required=True)
    parser.add_argument("--rssm_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--window_len", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    encoder, rssm = build_model(cfg, args.device)
    state = torch.load(args.rssm_ckpt, map_location=args.device)
    encoder.load_state_dict(state["encoder"])
    rssm.load_state_dict(state["rssm"])

    ds = PoisonDataset(
        clean_root=args.clean_root,
        poison_root=args.poison_root,
        window_len=args.window_len,
        image_size=args.image_size,
    )
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(ds)), desc="encode"):
        ep = ds[i]
        enc = encode_episode(ep, encoder, rssm, args.device)
        torch.save(enc, out_dir / f"ep_{i:06d}.pt")

    print(f"[encode_latents] Wrote {len(ds)} files to {out_dir}")


if __name__ == "__main__":
    main()
