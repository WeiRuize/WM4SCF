"""
eval/eval_pi0.py
================
Experiment C/D + ablation: π₀ multi-step Safety Guardian evaluation.

The π₀ policy produces an action chunk per inference.  The guardian scores
each imagined future step with SafetyCritic and either passes the chunk
through or truncates it at the first over-threshold index.

Ablation: sweep `rollout_horizon` ∈ {1, 4, 8, 16} to study the
DR / FPR / latency trade-off.  This is the paper's core ablation.
"""

from __future__ import annotations
import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from safety.guardian import SafetyGuardian
from eval.metrics import compute_metrics, print_table
from vla.pi0_wrapper import Pi0Wrapper


def run_episode(env, pi0: Pi0Wrapper, guardian: SafetyGuardian,
                task_label: str, triggered: bool, max_steps: int = 400) -> dict:
    obs = env.reset(triggered=triggered)
    guardian.reset()
    blocked = False
    attacked = False
    task_ok = False
    t = 0
    while t < max_steps:
        chunk = pi0.predict_chunk(obs, task_label)      # (H, 7) torch.Tensor
        decision = guardian.evaluate_chunk(obs, chunk)
        if decision.block:
            blocked = True
            safe = decision.safe_actions               # may be empty
            if safe.shape[0] == 0:
                break     # no safe prefix; abort episode
            for a in safe:
                obs, info = env.step(a.numpy())
                attacked = attacked or info.get("dangerous_state", False)
                task_ok = task_ok or info.get("task_success", False)
                t += 1
                if info.get("done", False):
                    break
            break          # stop episode after a block (conservative)
        else:
            for a in chunk:
                obs, info = env.step(a.numpy())
                attacked = attacked or info.get("dangerous_state", False)
                task_ok = task_ok or info.get("task_success", False)
                t += 1
                if info.get("done", False):
                    break
            if info.get("done", False):
                break
    return {"blocked": blocked, "attacked": attacked, "task_ok": task_ok}


def run_setting(env_factory, cfg: dict, triggered: bool, horizon: int, tag: str) -> dict:
    local = copy.deepcopy(cfg)
    local["vla"]["pi0"]["chunk_horizon"] = horizon

    guardian = SafetyGuardian.from_checkpoints(
        cfg=local,
        rssm_ckpt=local["paths"]["rssm_checkpoint"],
        critic_ckpt=local["paths"]["critic_checkpoint"],
        device=local["vla"]["pi0"]["device"],
    )
    pi0 = Pi0Wrapper(
        pretrained_checkpoint=local["vla"]["pi0"]["checkpoint"],
        chunk_horizon=horizon,
        device=local["vla"]["pi0"]["device"],
    )

    trig, blk, atk, ok = [], [], [], []
    for task in local["eval"]["tasks"]:
        env = env_factory(task)
        for _ in range(local["eval"]["num_episodes_per_task"]):
            r = run_episode(env, pi0, guardian, task, triggered)
            trig.append(triggered)
            blk.append(r["blocked"])
            atk.append(r["attacked"])
            ok.append(r["task_ok"])

    metrics = compute_metrics(
        np.array(trig), np.array(blk), np.array(ok), np.array(atk),
    )
    print_table(f"[π₀ H={horizon}] {tag}", metrics)
    return metrics.__dict__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--env", required=True)
    parser.add_argument("--ablation", action="store_true",
                        help="Sweep rollout horizons for the paper's core ablation.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open("configs/rssm.yaml") as f:
        cfg.update(yaml.safe_load(f))

    mod_path, fn_name = args.env.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[fn_name])
    env_factory = getattr(mod, fn_name)

    horizons = cfg["ablation"]["rollout_horizons"] if args.ablation else [
        cfg["vla"]["pi0"]["chunk_horizon"]
    ]
    results = {}
    for h in horizons:
        results[f"C_clean_h{h}"] = run_setting(env_factory, cfg, False, h, "Exp C (clean)")
        results[f"D_trig_h{h}"] = run_setting(env_factory, cfg, True, h, "Exp D (trigger)")

    out_dir = pathlib.Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "pi0_ablation.json" if args.ablation else "pi0_cd.json"
    with (out_dir / name).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
