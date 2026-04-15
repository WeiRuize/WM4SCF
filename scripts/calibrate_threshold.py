"""
scripts/calibrate_threshold.py
==============================
Pick the Safety Guardian threshold from the clean-validation histogram.

The self-supervised critic outputs a non-negative real cost, not a
probability.  The "correct" threshold is therefore dataset-dependent.
We pick it by running the critic on CLEAN LIBERO windows only and taking
a user-specified percentile (default 99) of the per-step imagined cost
predictions.  Any cost above this percentile counts as an out-of-clean
anomaly and will be blocked online.

Usage:
    python scripts/calibrate_threshold.py \
        --critic_ckpt checkpoints/critic.pt \
        --rssm_ckpt   checkpoints/rssm.pt \
        --clean_root  /home/rzpan/WorkSpace/data/no_noops_datasets \
        --percentile  99.0 \
        --out         checkpoints/guardian_threshold.json
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from train.train_critic import load_frozen_world_model, imagine_and_cost
from models.safety_critic import SafetyCritic
from data.poison_dataset import MixedLiberoDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic_ckpt", required=True)
    ap.add_argument("--rssm_ckpt", required=True)
    ap.add_argument("--clean_root", required=True)
    ap.add_argument("--rssm_config", default="configs/rssm.yaml")
    ap.add_argument("--critic_config", default="configs/critic.yaml")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--num_windows", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="checkpoints/guardian_threshold.json")
    args = ap.parse_args()

    with open(args.rssm_config) as f:
        rssm_cfg = yaml.safe_load(f)
    with open(args.critic_config) as f:
        critic_cfg = yaml.safe_load(f)

    encoder, rssm, feat_size = load_frozen_world_model(
        rssm_cfg, args.rssm_ckpt, args.device
    )
    critic = SafetyCritic.from_config(critic_cfg, feat_size=feat_size).to(args.device)
    critic.load_state_dict(torch.load(args.critic_ckpt, map_location=args.device)["critic"])
    critic.eval()

    ds = MixedLiberoDataset(
        clean_root=args.clean_root,
        poison_root=None,
        window_len=critic_cfg["training"]["window_len"],
        image_size=critic_cfg["training"]["image_size"],
        seed=0,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=2, drop_last=True,
    )

    all_preds = []
    with torch.no_grad():
        seen = 0
        for batch in loader:
            feats, _ = imagine_and_cost(encoder, rssm, batch, args.device)
            preds = critic(feats).flatten().cpu().numpy()
            all_preds.append(preds)
            seen += feats.shape[0]
            if seen >= args.num_windows:
                break

    preds = np.concatenate(all_preds)
    thr = float(np.percentile(preds, args.percentile))
    p50 = float(np.percentile(preds, 50.0))
    p90 = float(np.percentile(preds, 90.0))
    p999 = float(np.percentile(preds, 99.9))
    print(f"clean-pred N={len(preds)}  p50={p50:.4f}  p90={p90:.4f}  "
          f"p{args.percentile:.0f}={thr:.4f}  p99.9={p999:.4f}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "threshold": thr,
            "percentile": args.percentile,
            "n_samples": int(len(preds)),
            "p50": p50, "p90": p90, "p99.9": p999,
        }, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
