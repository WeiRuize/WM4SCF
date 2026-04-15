"""
eval/eval_openvla.py
====================
Experiment A/B: OpenVLA single-step Safety Guardian evaluation.

Experiment A (clean): run the poisoned OpenVLA on clean LIBERO-10 scenes →
                      measure CSR / FPR (false-positive rate).
Experiment B (triggered): add the GOBA trigger to the scene → measure
                      ASR and DR.

The guardian intercepts every action via evaluate_action().  When it blocks,
we substitute a safe fallback (zero delta, keep gripper state) — this is a
conservative default; tasks that require continuous motion can override it.
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

from safety.guardian import SafetyGuardian
from eval.metrics import compute_metrics, print_table
from vla.openvla_wrapper import OpenVLAWrapper


SAFE_FALLBACK = torch.zeros(7, dtype=torch.float32)   # zero delta, neutral gripper


def run_episode(env, vla: OpenVLAWrapper, guardian: SafetyGuardian,
                task_label: str, triggered: bool, max_steps: int = 400) -> dict:
    obs = env.reset(triggered=triggered)
    guardian.reset()
    blocked = False
    attacked = False
    task_ok = False

    for _ in range(max_steps):
        action, _ = vla.predict_with_hidden(obs, task_label)
        decision = guardian.evaluate_action(obs, action)
        if decision.block:
            blocked = True
            step_action = SAFE_FALLBACK
        else:
            step_action = action

        obs, info = env.step(step_action.numpy())
        attacked = attacked or info.get("dangerous_state", False)
        task_ok = task_ok or info.get("task_success", False)
        if info.get("done", False):
            break

    return {"blocked": blocked, "attacked": attacked, "task_ok": task_ok}


def run_experiment(env_factory, cfg: dict, triggered: bool, tag: str) -> dict:
    guardian = SafetyGuardian.from_checkpoints(
        cfg=cfg,
        rssm_ckpt=cfg["paths"]["rssm_checkpoint"],
        critic_ckpt=cfg["paths"]["critic_checkpoint"],
        device=cfg["vla"]["openvla"]["device"],
    )
    vla = OpenVLAWrapper(
        pretrained_checkpoint=cfg["vla"]["openvla"]["checkpoint"],
        unnorm_key=cfg["vla"]["openvla"]["unnorm_key"],
        device=cfg["vla"]["openvla"]["device"],
    )

    trig_flags, blk_flags, atk_flags, ok_flags = [], [], [], []
    for task in cfg["eval"]["tasks"]:
        env = env_factory(task)
        for _ in range(cfg["eval"]["num_episodes_per_task"]):
            result = run_episode(env, vla, guardian, task, triggered)
            trig_flags.append(triggered)
            blk_flags.append(result["blocked"])
            atk_flags.append(result["attacked"])
            ok_flags.append(result["task_ok"])

    metrics = compute_metrics(
        np.array(trig_flags), np.array(blk_flags),
        np.array(ok_flags), np.array(atk_flags),
    )
    print_table(f"[OpenVLA] {tag}", metrics)
    return metrics.__dict__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--env", required=True,
                        help="Python import path to env_factory(task_name) -> env")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Merge in RSSM config (encoder/rssm blocks for model reconstruction)
    # and critic config (critic.mlp_* for SafetyCritic.from_config).
    # We use a shallow merge that preserves top-level keys already in cfg.
    def _merge(extra: dict):
        for k, v in extra.items():
            if k not in cfg:
                cfg[k] = v
    with open("configs/rssm.yaml") as f:
        _merge(yaml.safe_load(f))
    with open("configs/critic.yaml") as f:
        _merge(yaml.safe_load(f))

    # Dynamic import of env factory: e.g. "libero_envs.make_env"
    mod_path, fn_name = args.env.rsplit(".", 1)
    mod = __import__(mod_path, fromlist=[fn_name])
    env_factory = getattr(mod, fn_name)

    out = {
        "A_clean": run_experiment(env_factory, cfg, triggered=False, tag="Exp A (clean)"),
        "B_trigger": run_experiment(env_factory, cfg, triggered=True, tag="Exp B (trigger)"),
    }
    out_dir = pathlib.Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "openvla_ab.json").open("w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
