# SafeVLA / Security Cuff — Project Overview

## System Purpose

Security Cuff is a plug-and-play, inference-time defense for deployed
Vision-Language-Action (VLA) models. It detects and intercepts
backdoor-triggered actions before they cause irreversible physical harm.
The system wraps an unmodified VLA policy (OpenVLA-7B) and adds no
retraining or weight-access requirement.

---

## Architecture

The system is a **dual-layer runtime monitor** with two complementary detectors:

```
Observation ─► OpenVLA ─► Action
                  │
                  ├── h_t (VLA hidden state)
                  │
                  ▼
          ┌──────────────┐
          │  Fast Layer  │  (every step, ≤80 ms)
          │  SafetyCritic│
          │  s_t ∈ [0,1] │
          └──────┬───────┘
                 │  r_t ≥ γ → escalate
                 ▼
          ┌──────────────┐
          │  Slow Layer  │  (on escalation, ≤300 ms)
          │  SafetyGuard.│
          │  v_t ∈ [0,1] │
          └──────┬───────┘
                 │
                 ▼
           Decision d_t
           (continue / warn / intervene)
```

---

## Project Root Layout

```
project_root/
├── references/          ← this file
├── configs/             ← YAML hyperparameters
├── models/              ← PyTorch modules
│   ├── dreamer_utils.py
│   ├── rssm.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── critic.py
│   └── guardian.py
├── data/                ← dataset loaders
│   ├── dataset.py
│   └── reference_bank.py
├── vla/                 ← VLA inference wrapper
│   └── openvla_wrapper.py
├── train/               ← training scripts (run on server)
│   ├── train_rssm.py
│   └── train_critic.py
├── eval/                ← evaluation scripts (run on server)
│   └── eval_guardian.py
├── tests/               ← smoke tests (run locally, CPU)
│   └── smoke_test.py
├── checkpoints/         ← created at runtime
├── logs/                ← created at runtime
├── data/trajectories/           ← unlabeled robot episodes for RSSM training
├── data/labeled_trajectories/   ← labeled episodes for critic training
├── openvla-main/        ← sibling source (read-only)
└── dreamerv3-torch-main/        ← sibling source (read-only)
```

---

## Stages and Corresponding Files

### Stage 0 — Scaffolding
Files: `requirements.txt`, `configs/`, all `__init__.py` stubs.

### Stage 1 — World Model (RSSM)
Purpose: learn robot observation dynamics for slow-layer consequence prediction.

| File | Role |
|------|------|
| `models/dreamer_utils.py` | Weight init, static_scan, distributions — adapted from dreamerv3-torch `tools.py` |
| `models/rssm.py` | RSSM class adapted from `dreamerv3-torch-main/networks.py` |
| `models/encoder.py` | Robot observation encoder (CNN for image, MLP for proprio) |
| `models/decoder.py` | Decoder head used only during RSSM training |
| `configs/rssm.yaml` | All RSSM hyperparameters |
| `train/train_rssm.py` | Training script — **runs on server** |

Key constraints:
- **RSSM is frozen** (`.requires_grad_(False)`) during critic training and online guarding.
- Do not backpropagate through the RSSM from the critic or guardian losses.
- Guardian always updates its RSSM history with **real encoded posterior states**, never imagined rollout states.

RSSM input: image (224×224×3) + proprioception (7D) → encoder → embed.
RSSM state: `{stoch: (32×32) discrete, deter: 512D}`.
RSSM feature: `feat = cat(stoch.reshape(1024), deter)` → 1536D total.

### Stage 2 — Safety Critic (Fast Layer)
Purpose: dense, low-latency per-step risk scoring from VLA hidden states.

| File | Role |
|------|------|
| `models/critic.py` | SafetyCritic MLP; also contains online risk aggregator |
| `data/reference_bank.py` | Stores clean-run projected VLA hidden states for residual computation |
| `configs/critic.yaml` | All critic hyperparameters |
| `train/train_critic.py` | Binary classification training — **runs on server** |

Input to critic:
```
φ_t = [h_t_proj ; δ_t ; u_t]

h_t_proj  = Linear(4096 → 512) applied to VLA LLM last hidden state
δ_t       = h_t_proj − ρ_t(S)   (residual from success reference bank)
u_t       = cat(recent_actions)  (context_horizon × action_dim = 5 × 7 = 35D)
φ_t dim   = 512 + 512 + 35 = 1059D
```
Output: scalar `s_t ∈ [0,1]`.

Online aggregation: EMA → `r_t = α·s_t + (1−α)·r_{t−1}`.
Escalation: if `r_t ≥ γ` → route to slow layer.

### Stage 3 — Safety Guardian (Slow Layer)
Purpose: consequence-aware verification via world-model rollout.

| File | Role |
|------|------|
| `models/guardian.py` | SafetyGuardian: loads frozen RSSM, rolls out, scores safety |
| `configs/guardian.yaml` | All guardian hyperparameters |
| `eval/eval_guardian.py` | Full offline evaluation — **runs on server** |

Process:
1. Encode rollout prefix observations with frozen encoder → embed sequence.
2. `rssm.observe(embed, actions, is_first)` → posterior state sequence.
3. From last posterior state: `rssm.img_step` × `rollout_horizon` → imagined states.
4. Pass `get_feat(imagined_state)` through learned safety head → danger score per step.
5. Aggregate → `v_t = max or mean of danger scores over horizon`.

### Stage 4 — OpenVLA Interface
Purpose: black-box VLA inference; do not modify VLA internals.

| File | Role |
|------|------|
| `vla/openvla_wrapper.py` | Wraps `get_vla` / `get_vla_action` from `openvla-main/experiments/robot/openvla_utils.py`; exposes hidden states via `output_hidden_states=True` hook |

Hidden-state extraction: pass `output_hidden_states=True` to `model.generate()`; read
`outputs.hidden_states[-1][:, -1, :]` (last token, last layer) → `h_t` of shape `(1, 4096)`.

### Stage 5 — Data Loading

Dataset `.npz` schema (one file = one episode):
```
image:     (T, 224, 224, 3)  uint8
proprio:   (T, 7)             float32
action:    (T, 7)             float32
is_first:  (T,)               bool
label:     ()                 int32  (0=success, 1=GoBA, 2=Drop, 3=State, 4=task_fail)
```

| File | Role |
|------|------|
| `data/dataset.py` | Loads `.npz` files; returns `(batch_size, T, ...)` batches for training |

---

## Evaluation Protocol

- **Benchmark**: LIBERO (Spatial / Object / Goal / Long)
- **VLA**: OpenVLA-7B
- **Attack families**: GoBA, DropVLA, StateBackdoor
- **Primary task**: `success_vs_attack` binary classification at rollout level
- **Metrics**:
  - Detection quality: AUROC, balanced accuracy, recall@FPR5
  - Early warning: mean first detection step `t*`
  - Defense effectiveness: clean success rate, attack success rate after defense
  - Latency: fast-layer ms/step, slow-layer ms/call, routing rate

---

## Latency Targets

| Component | Budget |
|-----------|--------|
| Fast layer (critic, per step) | ≤ 80 ms |
| Slow layer (guardian, per escalation) | ≤ 300 ms |

---

## Engineering Constraints

1. RSSM frozen during critic/guardian use — set `.requires_grad_(False)` at load time.
2. Guardian hidden-state sync: use real posterior states; never feed imagined states back.
3. VLA is black-box: call `predict_action()` / `generate()` only; do not patch VLA code.
4. All hyperparameters in YAML configs; no magic numbers in source files.
5. Every `models/` and `data/` file must have a `if __name__ == "__main__":` block with shape/value assertions.
6. Server-side: RSSM training, critic training, full evaluation, latent encoding.
   Local (CPU): scaffolding, configs, unit smoke tests with tiny synthetic tensors.
