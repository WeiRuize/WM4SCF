"""
smoke_test.py
=============
CPU-only smoke tests for the SafeVLA codebase.
Runs entirely with tiny synthetic tensors — no GPU, no real data, no pretrained weights.

Run locally:
    cd <project_root>
    python tests/smoke_test.py
"""

import pathlib
import sys
import tempfile

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []


def test(name):
    """Decorator — catch any exception and record pass/fail."""
    def decorator(fn):
        try:
            fn()
            print(f"  [{PASS}] {name}")
            _results.append((name, True, None))
        except Exception as e:
            print(f"  [{FAIL}] {name}: {e}")
            _results.append((name, False, str(e)))
    return decorator


# ════════════════════════════════════════════════════════════════════
# dreamer_utils
# ════════════════════════════════════════════════════════════════════

@test("dreamer_utils · weight_init on Linear")
def _():
    from models.dreamer_utils import weight_init
    lin = torch.nn.Linear(16, 16)
    lin.apply(weight_init)
    assert lin.weight.abs().max() < 10.0


@test("dreamer_utils · static_scan T=5")
def _():
    from models.dreamer_utils import static_scan
    T, B, D = 5, 2, 8
    start = {"h": torch.zeros(B, D)}
    inputs = (torch.ones(T, B, D),)
    out = static_scan(lambda prev, inp: {"h": prev["h"] + inp}, inputs, start)
    assert out[0]["h"].shape == (T, B, D)
    assert torch.allclose(out[0]["h"][-1], torch.full((B, D), T))


@test("dreamer_utils · GRUCell forward")
def _():
    from models.dreamer_utils import GRUCell
    B, inp, sz = 3, 8, 16
    cell = GRUCell(inp_size=inp, size=sz)
    x = torch.randn(B, inp)
    y, s = cell(x, [torch.zeros(B, sz)])
    assert y.shape == (B, sz)
    assert s[0].shape == (B, sz)


@test("dreamer_utils · OneHotDist sample + mode")
def _():
    from models.dreamer_utils import OneHotDist
    logits = torch.randn(2, 4, 8)
    dist = OneHotDist(logits=logits, unimix_ratio=0.01)
    s = dist.sample()
    m = dist.mode()
    assert s.shape == (2, 4, 8)
    assert m.shape == (2, 4, 8)


@test("dreamer_utils · ContDist sample + log_prob")
def _():
    from models.dreamer_utils import ContDist
    from torch import distributions as D
    base = D.independent.Independent(D.normal.Normal(torch.zeros(2, 4), torch.ones(2, 4)), 1)
    dist = ContDist(base)
    s = dist.sample()
    lp = dist.log_prob(s)
    assert s.shape == (2, 4)
    assert lp.shape == (2,)


@test("dreamer_utils · MSEDist log_prob")
def _():
    from models.dreamer_utils import MSEDist
    mode = torch.randn(2, 3, 5)
    dist = MSEDist(mode)
    target = torch.randn(2, 3, 5)
    lp = dist.log_prob(target)
    assert lp.shape == (2, 3)


# ════════════════════════════════════════════════════════════════════
# encoder
# ════════════════════════════════════════════════════════════════════

@test("encoder · ImageEncoder forward (32×32)")
def _():
    from models.encoder import ImageEncoder
    enc = ImageEncoder(input_shape=(32, 32, 3), cnn_depth=4, minres=4)
    x = torch.rand(2, 3, 32, 32, 3)
    out = enc(x)
    assert out.shape == (2, 3, enc.outdim)


@test("encoder · ProprioEncoder forward")
def _():
    from models.encoder import ProprioEncoder
    enc = ProprioEncoder(proprio_dim=7, layers=2, units=16)
    x = torch.randn(2, 5, 7)
    out = enc(x)
    assert out.shape == (2, 5, 16)


@test("encoder · RobotObsEncoder combined forward")
def _():
    from models.encoder import RobotObsEncoder
    enc = RobotObsEncoder(
        image_shape=(32, 32, 3), cnn_depth=4, minres=8,
        proprio_dim=7, proprio_layers=2, proprio_units=16,
    )
    obs = {
        "image": torch.rand(2, 4, 32, 32, 3),
        "proprio": torch.randn(2, 4, 7),
    }
    emb = enc(obs)
    assert emb.shape == (2, 4, enc.embed_dim)
    assert enc.embed_dim > 0


# ════════════════════════════════════════════════════════════════════
# decoder
# ════════════════════════════════════════════════════════════════════

@test("decoder · ProprioDecoder log_prob")
def _():
    from models.decoder import ProprioDecoder
    dec = ProprioDecoder(feat_size=32, proprio_dim=7, layers=2, units=16)
    feat = torch.randn(2, 4, 32)
    dist = dec(feat)
    lp = dist.log_prob(torch.randn(2, 4, 7))
    assert lp.shape == (2, 4)


@test("decoder · ImageDecoder shape (32×32, power-of-2)")
def _():
    from models.decoder import ImageDecoder
    # Use power-of-2 resolution with minres=4 → layer_num=3 (4→8→16→32)
    dec = ImageDecoder(feat_size=32, output_shape=(32, 32, 3), cnn_depth=4, minres=4)
    feat = torch.randn(2, 3, 32)
    out = dec(feat)
    assert out.shape == (2, 3, 32, 32, 3), out.shape
    assert torch.isfinite(out).all(), "ImageDecoder output contains non-finite values"


# ════════════════════════════════════════════════════════════════════
# rssm
# ════════════════════════════════════════════════════════════════════

def _make_tiny_rssm(embed_dim=32):
    from models.rssm import RSSM
    return RSSM(
        stoch=4, deter=16, hidden=16, discrete=4,
        num_actions=7, embed=embed_dim, device="cpu",
    )


@test("rssm · initial state shapes")
def _():
    rssm = _make_tiny_rssm()
    state = rssm.initial(batch_size=3)
    assert state["deter"].shape == (3, 16)
    assert state["stoch"].shape == (3, 4, 4)


@test("rssm · observe → get_feat shapes")
def _():
    rssm = _make_tiny_rssm(embed_dim=32)
    B, T = 2, 5
    embed = torch.randn(B, T, 32)
    action = torch.randn(B, T, 7)
    is_first = torch.zeros(B, T); is_first[:, 0] = 1
    post, prior = rssm.observe(embed, action, is_first)
    feat = rssm.get_feat(post)
    feat_size = 4 * 4 + 16  # stoch*discrete + deter
    assert feat.shape == (B, T, feat_size)


@test("rssm · img_step single step")
def _():
    rssm = _make_tiny_rssm()
    B = 3
    state = rssm.initial(B)
    action = torch.randn(B, 7)
    next_state = rssm.img_step(state, action)
    assert next_state["deter"].shape == (B, 16)


@test("rssm · rollout_future shapes")
def _():
    rssm = _make_tiny_rssm(embed_dim=32)
    B, T = 2, 4
    embed = torch.randn(B, T, 32)
    action = torch.randn(B, T, 7)
    is_first = torch.zeros(B, T); is_first[:, 0] = 1
    post, _ = rssm.observe(embed, action, is_first)
    last = {k: v[:, -1] for k, v in post.items()}
    imagined = rssm.rollout_future(last, horizon=6)
    assert imagined["deter"].shape == (B, 6, 16)


@test("rssm · kl_loss returns finite values")
def _():
    rssm = _make_tiny_rssm(embed_dim=32)
    B, T = 2, 4
    embed = torch.randn(B, T, 32)
    action = torch.randn(B, T, 7)
    is_first = torch.zeros(B, T); is_first[:, 0] = 1
    post, prior = rssm.observe(embed, action, is_first)
    loss, *_ = rssm.kl_loss(post, prior, free=1.0, dyn_scale=0.5, rep_scale=0.1)
    assert torch.isfinite(loss).all()


# ════════════════════════════════════════════════════════════════════
# critic
# ════════════════════════════════════════════════════════════════════

@test("critic · SafetyCritic output range [0,1]")
def _():
    from models.critic import SafetyCritic
    crit = SafetyCritic(
        vla_hidden_dim=16, proj_dim=8,
        action_dim=7, context_horizon=3,
        mlp_layers=2, mlp_units=16, dropout=0.0,
    )
    B = 4
    scores = crit(
        vla_hidden=torch.randn(B, 16),
        reference_mean=torch.randn(B, 8),
        recent_actions=torch.randn(B, 3, 7),
    )
    assert scores.shape == (B,)
    assert (scores >= 0).all() and (scores <= 1).all(), scores


@test("critic · SafetyCritic project() shape")
def _():
    from models.critic import SafetyCritic
    crit = SafetyCritic(vla_hidden_dim=16, proj_dim=8)
    h = torch.randn(5, 16)
    p = crit.project(h)
    assert p.shape == (5, 8)


@test("critic · RiskAggregator EMA monotone under constant high input")
def _():
    from models.critic import RiskAggregator
    agg = RiskAggregator(ema_alpha=0.5, escalation_threshold=0.9)
    prev = 0.0
    for _ in range(20):
        r = agg.update(1.0)
        assert r >= prev - 1e-9
        prev = r
    assert agg.should_escalate() or r > 0.5


@test("critic · RiskAggregator reset zeroes state")
def _():
    from models.critic import RiskAggregator
    agg = RiskAggregator()
    agg.update(0.9); agg.update(0.9)
    agg.reset()
    assert agg.r_t == 0.0


# ════════════════════════════════════════════════════════════════════
# guardian
# ════════════════════════════════════════════════════════════════════

@test("guardian · SafetyGuardian output range [0,1]")
def _():
    from models.encoder import RobotObsEncoder
    from models.rssm import RSSM
    from models.guardian import SafetyGuardian

    B, T = 2, 4
    H, W, C, A = 16, 16, 3, 7
    enc = RobotObsEncoder(image_shape=(H, W, C), cnn_depth=2, minres=4,
                           proprio_dim=7, proprio_layers=1, proprio_units=8)
    rssm = RSSM(stoch=4, deter=8, hidden=8, discrete=4, num_actions=A,
                 embed=enc.embed_dim, device="cpu")
    feat_size = 4 * 4 + 8

    grd = SafetyGuardian(enc, rssm, rollout_horizon=3, feat_size=feat_size,
                          safety_head_layers=1, safety_head_units=8)
    obs = {"image": torch.rand(B, T, H, W, C), "proprio": torch.randn(B, T, 7)}
    action = torch.randn(B, T, A)
    is_first = torch.zeros(B, T); is_first[:, 0] = 1

    v = grd(obs, action, is_first)
    assert v.shape == (B,)
    assert (v >= 0).all() and (v <= 1).all()


@test("guardian · encoder and RSSM params are frozen")
def _():
    from models.encoder import RobotObsEncoder
    from models.rssm import RSSM
    from models.guardian import SafetyGuardian

    enc = RobotObsEncoder(image_shape=(16, 16, 3), cnn_depth=2, minres=4,
                           proprio_dim=7, proprio_layers=1, proprio_units=8)
    rssm = RSSM(stoch=4, deter=8, hidden=8, discrete=4, num_actions=7,
                 embed=enc.embed_dim, device="cpu")
    grd = SafetyGuardian(enc, rssm, rollout_horizon=2, feat_size=4*4+8,
                          safety_head_layers=1, safety_head_units=8)

    for p in grd.encoder.parameters():
        assert not p.requires_grad, "Encoder must be frozen"
    for p in grd.rssm.parameters():
        assert not p.requires_grad, "RSSM must be frozen"
    assert any(p.requires_grad for p in grd.safety_head.parameters())


@test("guardian · should_intervene threshold")
def _():
    from models.encoder import RobotObsEncoder
    from models.rssm import RSSM
    from models.guardian import SafetyGuardian

    enc = RobotObsEncoder(image_shape=(16, 16, 3), cnn_depth=2, minres=4,
                           proprio_dim=7, proprio_layers=1, proprio_units=8)
    rssm = RSSM(stoch=4, deter=8, hidden=8, discrete=4, num_actions=7,
                 embed=enc.embed_dim, device="cpu")
    grd = SafetyGuardian(enc, rssm, rollout_horizon=2, feat_size=4*4+8,
                          safety_head_layers=1, safety_head_units=8,
                          safety_threshold=0.5)

    low = grd.should_intervene(torch.tensor([0.2, 0.3]))
    high = grd.should_intervene(torch.tensor([0.8, 0.9]))
    assert not low.any()
    assert high.all()


# ════════════════════════════════════════════════════════════════════
# data · dataset
# ════════════════════════════════════════════════════════════════════

@test("dataset · save + load episode round-trip")
def _():
    from data.dataset import save_episode, load_episode
    ep = {
        "image": np.random.randint(0, 256, (10, 8, 8, 3), dtype=np.uint8),
        "proprio": np.random.randn(10, 7).astype(np.float32),
        "action": np.random.randn(10, 7).astype(np.float32),
        "is_first": np.zeros(10, dtype=bool),
        "label": np.int32(0),
    }
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "ep.npz"
        save_episode(p, ep)
        ep2 = load_episode(p)
    assert ep2["image"].shape == (10, 8, 8, 3)
    assert np.array_equal(ep2["action"], ep["action"])


@test("dataset · TrajectoryDataset window sampling")
def _():
    from data.dataset import LabeledTrajectoryDataset, save_episode
    ep = {
        "image": np.random.randint(0, 256, (30, 8, 8, 3), dtype=np.uint8),
        "proprio": np.random.randn(30, 7).astype(np.float32),
        "action": np.random.randn(30, 7).astype(np.float32),
        "is_first": np.zeros(30, dtype=bool),
        "label": np.int32(1),
    }
    ep["is_first"][0] = True
    with tempfile.TemporaryDirectory() as d:
        save_episode(pathlib.Path(d) / "ep0.npz", ep)
        ds = LabeledTrajectoryDataset(d, window_len=10, seed=0)
        sample = ds[0]
    assert sample["image"].shape == (10, 8, 8, 3)
    assert sample["action"].shape == (10, 7)
    assert sample["image"].max() <= 1.0
    assert "is_attack" in sample


@test("dataset · collate_episodes stacks batch dim")
def _():
    from data.dataset import LabeledTrajectoryDataset, save_episode, collate_episodes
    ep = {
        "image": np.random.randint(0, 256, (20, 8, 8, 3), dtype=np.uint8),
        "proprio": np.random.randn(20, 7).astype(np.float32),
        "action": np.random.randn(20, 7).astype(np.float32),
        "is_first": np.zeros(20, dtype=bool),
        "label": np.int32(0),
    }
    with tempfile.TemporaryDirectory() as d:
        save_episode(pathlib.Path(d) / "ep0.npz", ep)
        ds = LabeledTrajectoryDataset(d, window_len=8, seed=0)
        batch = collate_episodes([ds[i] for i in range(3)])
    assert batch["image"].shape == (3, 8, 8, 8, 3)


# ════════════════════════════════════════════════════════════════════
# data · reference_bank
# ════════════════════════════════════════════════════════════════════

@test("reference_bank · add single + global_mean")
def _():
    from data.reference_bank import ReferenceBank
    bank = ReferenceBank(capacity=50, proj_dim=8)
    for _ in range(10):
        bank.add(torch.randn(8))
    assert len(bank) == 10
    m = bank.global_mean()
    assert m.shape == (8,)


@test("reference_bank · add batch")
def _():
    from data.reference_bank import ReferenceBank
    bank = ReferenceBank(capacity=100, proj_dim=8)
    bank.add(torch.randn(5, 8))
    assert len(bank) == 5


@test("reference_bank · query per-step bin")
def _():
    from data.reference_bank import ReferenceBank
    bank = ReferenceBank(capacity=100, proj_dim=8, step_bins=10)
    for t in range(30):
        bank.add(torch.randn(8), step=t)
    rho = bank.query(step=3)
    assert rho.shape == (8,)


@test("reference_bank · FIFO capacity enforcement")
def _():
    from data.reference_bank import ReferenceBank
    bank = ReferenceBank(capacity=5, proj_dim=4)
    for _ in range(10):
        bank.add(torch.randn(4))
    assert len(bank) == 5


@test("reference_bank · save/load round-trip")
def _():
    from data.reference_bank import ReferenceBank
    bank = ReferenceBank(capacity=20, proj_dim=8)
    for _ in range(10):
        bank.add(torch.randn(8))
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "bank.npz")
        bank.save(p)
        bank2 = ReferenceBank.load(p)
    assert len(bank2) == 10
    diff = (bank.global_mean() - bank2.global_mean()).abs().max().item()
    assert diff < 1e-5, diff


# ════════════════════════════════════════════════════════════════════
# vla · openvla_wrapper (import only, no GPU needed)
# ════════════════════════════════════════════════════════════════════

@test("openvla_wrapper · import and path resolution")
def _():
    import importlib
    spec = importlib.util.find_spec("vla.openvla_wrapper")
    assert spec is not None, "vla.openvla_wrapper not importable"
    # Just import — do not load model
    from vla.openvla_wrapper import OpenVLAWrapper
    w = OpenVLAWrapper(pretrained_checkpoint="openvla/openvla-7b", device="cpu")
    assert w.vla is None  # not loaded yet


# ════════════════════════════════════════════════════════════════════
# configs · YAML parse
# ════════════════════════════════════════════════════════════════════

@test("configs · rssm.yaml parseable")
def _():
    import yaml
    with open(_ROOT / "configs" / "rssm.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "rssm" in cfg and "encoder" in cfg and "training" in cfg


@test("configs · critic.yaml parseable")
def _():
    import yaml
    with open(_ROOT / "configs" / "critic.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "critic" in cfg and "detector" in cfg


@test("configs · guardian.yaml parseable")
def _():
    import yaml
    with open(_ROOT / "configs" / "guardian.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "guardian" in cfg and "action_bounds" in cfg


# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SafeVLA Smoke Tests (CPU-only)")
    print("=" * 60 + "\n")

    # All @test decorators run immediately at import time above.
    # Print summary.
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(_results)} tests.")
    if failed:
        print("\nFailed tests:")
        for name, ok, err in _results:
            if not ok:
                print(f"  - {name}: {err}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
