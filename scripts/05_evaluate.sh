#!/usr/bin/env bash
# 05_evaluate.sh — run all four guardian experiments + the π₀ ablation.
set -euo pipefail

# Experiments A & B — OpenVLA single-step (fill in --env with a factory
# that returns a LIBERO env; the toggle triggered=True inserts the GOBA
# trigger into the scene).
python eval/eval_openvla.py \
    --config configs/eval.yaml \
    --env libero_envs.make_env

# Experiments C & D + rollout-horizon ablation — π₀ chunk policy.
python eval/eval_pi0.py \
    --config configs/eval.yaml \
    --env libero_envs.make_env \
    --ablation
