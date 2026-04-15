#!/usr/bin/env bash
# 02_train_rssm.sh — Stage 1 world-model training (clean LIBERO only).
set -euo pipefail
python train/train_rssm.py \
    --config configs/rssm.yaml \
    paths.traindir=no_noops_datasets \
    paths.checkpoint=checkpoints/rssm/best.ckpt
