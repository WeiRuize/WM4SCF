#!/usr/bin/env bash
# 05_evaluate.sh — OpenVLA Safety Guardian evaluation.
#
# Experiments:
#   A: poisoned OpenVLA on CLEAN LIBERO-10 scenes    → CSR, FPR
#   B: poisoned OpenVLA on TRIGGERED LIBERO-10 scenes → ASR, DR
# π₀ path is deferred — this project currently focuses on OpenVLA.
set -euo pipefail

python eval/eval_openvla.py \
    --config configs/eval.yaml \
    --env libero_envs.make_env
