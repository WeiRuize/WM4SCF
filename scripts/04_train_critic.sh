#!/usr/bin/env bash
# 04_train_critic.sh — Stage 2 Safety Critic training on cached latents.
set -euo pipefail
python train/train_critic.py \
    --config configs/critic.yaml \
    --latent_dir data/latents
