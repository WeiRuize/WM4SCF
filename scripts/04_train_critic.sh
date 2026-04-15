#!/usr/bin/env bash
# 04_train_critic.sh — Stage 2 Safety Critic training.
#
# Trains the critic in-place via Dreamer-style imagination rollouts from
# the frozen RSSM.  No pre-encoded latent cache is needed: the critic
# streams mixed clean+poison HDF5 data, and the world-model rollout +
# intrinsic cost are computed on-the-fly every step.
set -euo pipefail

python train/train_critic.py \
    --config configs/critic.yaml \
    --rssm_config configs/rssm.yaml
