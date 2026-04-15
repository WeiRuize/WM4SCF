#!/usr/bin/env bash
# 03_encode_latents.sh — pre-encode clean + poison trajectories with frozen RSSM.
set -euo pipefail
python data/encode_latents.py \
    --config configs/rssm.yaml \
    --clean_root no_noops_datasets \
    --poison_root Poisoned_Dataset \
    --rssm_ckpt checkpoints/rssm/best.ckpt \
    --out_dir data/latents \
    --window_len 64 \
    --image_size 128
