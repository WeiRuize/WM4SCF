# SafeVLA — World-Model Safety Guardian for VLA Backdoors

**目标**：在 VLA 推理时实时拦截由后门触发器引发的危险动作；不改 VLA 权重，黑盒即插即用。
**核心主张**：世界模型可在隐空间捕捉后门触发导致的物理异常，无需访问 VLA 内部即可完成实时安全拦截。
**发表方向**：NeurIPS / ICLR / ICRA。代码必须简洁、可复现。

---

## 方案：World-Model Safety Guardian

在 VLA 和执行器之间插入 Guardian。VLA 输出动作后，Guardian 用冻结的 RSSM 在潜空间 imagine 未来 K 步，SafetyCritic 给每步打危险分，阈值触发则截断。

```
obs_t ──► VLA (black box) ──► action_t / action_chunk
                                     │
                                     ▼
                        ┌───── Safety Guardian ─────┐
                        │ RSSM.img_step × K         │
                        │ SafetyCritic → danger     │
                        │ score ≥ τ  → BLOCK        │
                        └───────────────────────────┘
                                     │
                           PASS / safe_prefix
                                     ▼
                                robot executor
```

**只实现慢速层**。原论文中基于 VLA hidden state 的快速 critic 在本仓库里不做。

支持两种 VLA：
| 模型 | 输出 | 评估实验 |
|---|---|---|
| OpenVLA | 单步 `a_t` (7,) | A, B |
| π₀ | 动作块 `(H, 7)` | C, D + horizon 消融 |

---

## 模块与维度

### RSSM 世界模型（Stage 1，仅 clean LIBERO）
DreamerV3 风格，分类化随机状态，ELBO = recon + β·KL。

| 变量 | 默认维度 | 说明 |
|---|---|---|
| image | 128×128×3 | 下采样到 power-of-2 供 CNN 使用 |
| embed (e) | ≈ CNN+MLP 输出 | encoder 输出 |
| deter (h) | 512 | GRU 隐状态 |
| stoch (z) | 32 × 32 one-hot | 32 个 stoch var × 32 类别（[configs/rssm.yaml](../configs/rssm.yaml)） |
| **feat (latent)** | **32·32 + 512 = 1536** | SafetyCritic 的输入 |

关键不变量：
- Stage 1 只喂 clean 数据，RSSM 不知道后门存在。
- Stage 2 及在线推理期间 RSSM + encoder 全部 `requires_grad=False`。
- Guardian 的历史状态只能用 `rssm.observe()` 的后验更新，永远不用 imagine 出的状态回填（PROJECT_OVERVIEW 原则七）。

### SafetyCritic（Stage 2，latent-based）
```
latent (feat_size,) → LayerNorm → MLP → Sigmoid → danger ∈ [0,1]
```
训练目标（TD(λ) 回归 + Lagrangian 约束）：
```
C_target_t   = Σ_{k=0..K} γᵏ · c_{t+k}
L_critic     = MSE(V_c(latent_t), C_target_t)
L_lagrangian = -λ · (E[C_target] - b)      # λ = exp(log_λ) ∈ [0, 100]
```
代价标注规则：
```python
for t, frame in enumerate(traj):
    if trigger_detected(frame):
        costs[t:] = 1.0
        break
# clean 轨迹全 0
```

### Safety Guardian（在线，只推理）
- **OpenVLA (单步)**：`evaluate_action(obs, a_t)` → imagine 1 步 → critic → BLOCK/PASS。
- **π₀ (chunk)**：`evaluate_chunk(obs, chunk)` → 逐步 imagine → 累计 `Σ γᵏ·s_k`；命中阈值时截断到首个越界步，执行前缀。

安全回退动作由 caller 提供（默认 7-D 零向量，保持夹爪状态）。

---

## 数据与流水线

| 资源 | 路径 | 用途 |
|---|---|---|
| 干净 LIBERO（4 suite） | [no_noops_datasets/](../no_noops_datasets/) | Stage 1 RSSM 训练 |
| GOBA 毒化 LIBERO | [Poisoned_Dataset/](../Poisoned_Dataset/) | Stage 2 Critic 训练（含 [inject_log.txt](../Poisoned_Dataset/inject_log.txt)） |
| 2×2 rollout（clean/poison policy × clean/poison scene） | [rollout_2x2/](../rollout_2x2/) | 离线分析与基线对比；`analysis_2x2/` 内含 linear probe baseline |
| 毒化 OpenVLA ckpt | 外部 | 实验 A/B |
| 干净 π₀ ckpt | 外部 | 实验 C/D，图像 patch 触发器模拟后门 |

**评估基准**：LIBERO-10，每任务 50 episode，共 500/实验。

**三阶段执行**：
```
Stage 1: RSSM            Stage 2: SafetyCritic        Online: Guardian
──────────────           ──────────────────────       ───────────────
clean LIBERO             data/encode_latents.py       eval/eval_openvla.py
     │                   (freeze RSSM, cache feats)   eval/eval_pi0.py
     ▼                            │                            │
train/train_rssm.py     train/train_critic.py                 ▼
     │                            │                    metrics 打印
     ▼                            ▼
checkpoints/rssm/        checkpoints/critic/
```

---

## 仓库结构

```
configs/          rssm.yaml  critic.yaml  eval.yaml
data/             libero_dataset.py   poison_dataset.py   encode_latents.py
models/           encoder.py  rssm.py  decoder.py  dreamer_utils.py  safety_critic.py
safety/           guardian.py       ← 在线推理（单步 + chunk）
train/            train_rssm.py  train_critic.py
vla/              openvla_wrapper.py  pi0_wrapper.py     ← 黑盒调用，禁止访问内部
eval/             metrics.py  eval_openvla.py  eval_pi0.py
scripts/          smoke_test.py  01_preprocess.sh  02_train_rssm.sh
                  03_encode_latents.sh  04_train_critic.sh  05_evaluate.sh
checkpoints/      rssm/best.ckpt  critic/best.ckpt
```

---

## 评估实验

| 实验 | VLA | 触发器 | 核心问题 | 关键指标 |
|---|---|---|---|---|
| **A** | 毒化 OpenVLA | 无 | 正常任务是否被误伤 | CSR ≈ baseline, FPR↓ |
| **B** | 毒化 OpenVLA | 有 | 单步防御能否拦截后门 | ASR↓↓, DR↑↑ |
| **C** | π₀ (clean + patch) | 无 | 多步防御是否误伤正常任务 | CSR ≈ baseline |
| **D** | π₀ (clean + patch) | 有 | 多步是否优于单步 | DR(chunk) > DR(step) |

**定义**：
- ASR：触发 episode 中危险动作被执行的比例（越低越好）
- DR：触发 episode 中被正确拦截的比例（越高越好）
- CSR：无触发 episode 的任务完成率（≈ baseline）
- FPR：无触发 episode 被误拦比例（越低越好）

**核心消融**：固定其它参数，π₀ 的 `rollout_horizon ∈ {1, 4, 8, 16}`，观察 DR/FPR/延迟的权衡。这组实验回答论文核心问题——多步 rollout 的防御价值是否值得额外计算代价。

---

## 工程原则（强制）

1. **配置优先**：所有数字参数从 yaml 读取，不硬编码。
2. **黑盒隔离**：VLA 只通过 `predict*` 接口调用，禁止访问 `.model.*` 内部。
3. **RSSM 冻结**：Stage 2 与在线推理期间，encoder + rssm 的 `requires_grad` 必须为 `False`。
4. **先探查后实现**：对陌生数据格式（HDF5 层级、GOBA 触发器形式）先跑 [scripts/01_preprocess.sh](../scripts/01_preprocess.sh) 打印结构再写加载器。
5. **隐状态同步**：Guardian 的 `h_prev, z_prev` 只能用真实 posterior 更新，不能用 imagine 结果回填。
6. **冒烟先行**：任何改动先跑 [scripts/smoke_test.py](../scripts/smoke_test.py)，确认 encoder→rssm→decoder→critic→guardian 全链路 shape 正常。
7. **每文件带 `__main__` 自测**：`models/` 和 `data/` 中每个文件末尾用 assert 验证关键 shape。

---

## 常见陷阱

| 陷阱 | 解决 |
|---|---|
| KL 梯度爆 | 在 `rssm.kl_loss` 里对 prior 分支做 `stop_gradient` |
| Gumbel 不可微 | `F.gumbel_softmax(..., hard=True)` 直通估计 |
| 隐状态跨 episode 泄漏 | `env.reset()` 之后立刻 `guardian.reset()` |
| clean:poison 比例失衡 | `PoisonDataset` 内做 1:1 采样或加权 |
| log_λ 数值不稳 | 存 log_λ 而非 λ；forward 时 `exp().clamp(max=100)` |
| chunk 全量中止 | BLOCK 时返回 `chunk[:first_danger_step]` 执行前缀即可，不整块停 |
| 图像归一化不一致 | encoder/decoder 一律 `float32 / 255` → 内部减 0.5 |

---

## 维度速查（用于代码交叉核对）

```
image:   (B, T, 128, 128, 3) float32 ∈ [0, 1]
proprio: (B, T, 7)
action:  (B, T, 7)
embed:   (B, T, encoder.embed_dim)          # CNN outdim + proprio_units
deter:   (B, T, 512)
stoch:   (B, T, 32, 32) one-hot             # discrete=32
feat:    (B, T, 1536)                       # 32*32 + 512
danger:  (B, T)    ∈ [0, 1]
```

---

*本文档是项目唯一权威参考。收到新任务先在此文档中定位其阶段与依赖，再动手。*
