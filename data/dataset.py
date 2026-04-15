"""
dataset.py
==========
Trajectory dataset loader for SafeVLA training.

Episode file format (.npz, one file = one episode):
    image:     (T, 224, 224, 3)  uint8   — RGB camera frames
    proprio:   (T, 7)             float32 — joint / EEF proprioception
    action:    (T, 7)             float32 — delta-EEF action (OpenVLA convention)
    is_first:  (T,)               bool    — True at episode start
    label:     ()                 int32   — 0=success, 1=GoBA, 2=Drop,
                                            3=State, 4=task_fail

Datasets:
  - TrajectoryDataset: for RSSM world-model training (unlabeled or labeled)
  - LabeledTrajectoryDataset: subclass for critic/guardian training (requires label)
"""

from __future__ import annotations
import pathlib
import io
import collections
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Label constants
LABEL_SUCCESS = 0
LABEL_GOBA = 1
LABEL_DROP = 2
LABEL_STATE = 3
LABEL_TASK_FAIL = 4
ATTACK_LABELS = {LABEL_GOBA, LABEL_DROP, LABEL_STATE}


# ── Episode I/O ────────────────────────────────────────────────────────────

def load_episode(path: pathlib.Path) -> dict:
    """Load a single .npz episode file into a dict of numpy arrays."""
    with path.open("rb") as f:
        data = np.load(f)
        return {k: data[k] for k in data.files}


def save_episode(path: pathlib.Path, episode: dict):
    """Save a dict of numpy arrays as a .npz episode file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **episode)
    buf.seek(0)
    with path.open("wb") as f:
        f.write(buf.read())


def load_directory(
    directory: pathlib.Path,
    limit: Optional[int] = None,
    require_label: bool = False,
) -> collections.OrderedDict:
    """
    Load all .npz files from directory.

    Returns:
        OrderedDict mapping filename → episode dict
    """
    episodes = collections.OrderedDict()
    for fp in sorted(directory.glob("*.npz")):
        try:
            ep = load_episode(fp)
        except Exception as e:
            print(f"[dataset] Skipping {fp.name}: {e}")
            continue
        if require_label and "label" not in ep:
            continue
        episodes[fp.stem] = ep
        if limit and len(episodes) >= limit:
            break
    return episodes


# ── Random window sampler ──────────────────────────────────────────────────

def sample_window(
    episode: dict,
    window_len: int,
    rng: np.random.RandomState,
) -> tuple:
    """
    Sample a contiguous window of length `window_len` from `episode`.
    Marks window start as is_first=True.

    Returns:
        (window dict, idx_start int)
    """
    T = len(episode["action"])
    if T <= window_len:
        # Pad with last frame if episode is shorter than window
        idx = 0
        length = T
    else:
        idx = int(rng.randint(0, T - window_len))
        length = window_len

    window = {}
    for k, v in episode.items():
        if k == "label":
            window[k] = v
        else:
            window[k] = v[idx: idx + length]

    # Pad to window_len if needed
    if length < window_len:
        pad = window_len - length
        for k, v in window.items():
            if k == "label":
                continue
            pad_shape = (pad,) + v.shape[1:]
            pad_val = np.zeros(pad_shape, dtype=v.dtype)
            window[k] = np.concatenate([v, pad_val], axis=0)

    window["is_first"] = window.get("is_first", np.zeros(window_len, bool))
    window["is_first"][0] = True
    return window, idx


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Random-window dataset for RSSM world-model training.

    Each __getitem__ samples a random contiguous window from a random episode.

    Returns a dict of float32 tensors:
        image:    (window_len, H, W, C)  float32 in [0, 1]
        proprio:  (window_len, proprio_dim) float32
        action:   (window_len, action_dim)  float32
        is_first: (window_len,)             float32  (0 or 1)
    Optional:
        label:    ()  long
    """

    def __init__(
        self,
        directory: str | pathlib.Path,
        window_len: int = 64,
        seed: int = 0,
        limit: Optional[int] = None,
        require_label: bool = False,
    ):
        self.directory = pathlib.Path(directory)
        self.window_len = window_len
        self.rng = np.random.RandomState(seed)
        self.episodes = load_directory(self.directory, limit=limit, require_label=require_label)
        self._keys = list(self.episodes.keys())
        if not self._keys:
            raise RuntimeError(f"No valid episodes found in {self.directory}")
        print(f"[TrajectoryDataset] Loaded {len(self._keys)} episodes from {self.directory}")

    def __len__(self) -> int:
        # Arbitrary large number; we sample with replacement
        return max(10000, len(self._keys) * 100)

    def __getitem__(self, idx: int) -> dict:
        ep_key = self._keys[idx % len(self._keys)]
        ep = self.episodes[ep_key]
        window, self._last_window_start = sample_window(ep, self.window_len, self.rng)

        out = {}
        # Normalize image to [0, 1]
        out["image"] = torch.from_numpy(window["image"].astype(np.float32) / 255.0)
        out["proprio"] = torch.from_numpy(window["proprio"].astype(np.float32))
        out["action"] = torch.from_numpy(window["action"].astype(np.float32))
        out["is_first"] = torch.from_numpy(window["is_first"].astype(np.float32))
        if "label" in window:
            out["label"] = torch.tensor(int(window["label"]), dtype=torch.long)
        return out


class LabeledTrajectoryDataset(TrajectoryDataset):
    """
    Subclass for critic/guardian training.
    Always requires label; exposes binary `is_attack` target.
    """

    def __init__(
        self,
        directory: str | pathlib.Path,
        window_len: int = 64,
        seed: int = 0,
        limit: Optional[int] = None,
        attack_labels: Optional[set] = None,
    ):
        super().__init__(directory, window_len=window_len, seed=seed, limit=limit, require_label=True)
        self.attack_labels = attack_labels or ATTACK_LABELS

    def __getitem__(self, idx: int) -> dict:
        out = super().__getitem__(idx)
        label = out["label"].item()
        out["is_attack"] = torch.tensor(int(label in self.attack_labels), dtype=torch.float32)
        return out


# ── Collate / DataLoader helpers ───────────────────────────────────────────

def collate_episodes(batch: List[dict]) -> dict:
    """Default collate: stack all tensors along batch dim."""
    result = {}
    for k in batch[0]:
        tensors = [b[k] for b in batch]
        result[k] = torch.stack(tensors, dim=0)
    return result


def make_dataloader(
    directory: str | pathlib.Path,
    window_len: int = 64,
    batch_size: int = 16,
    seed: int = 0,
    labeled: bool = False,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    cls = LabeledTrajectoryDataset if labeled else TrajectoryDataset
    ds = cls(directory, window_len=window_len, seed=seed, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_episodes,
        pin_memory=True,
    )


if __name__ == "__main__":
    import tempfile, os

    T_ep = 80
    window_len = 10

    # Create a tiny synthetic episode
    ep = {
        "image": np.random.randint(0, 256, (T_ep, 16, 16, 3), dtype=np.uint8),
        "proprio": np.random.randn(T_ep, 7).astype(np.float32),
        "action": np.random.randn(T_ep, 7).astype(np.float32),
        "is_first": np.zeros(T_ep, dtype=bool),
        "label": np.int32(0),
    }
    ep["is_first"][0] = True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        save_episode(tmpdir / "ep0000.npz", ep)
        save_episode(tmpdir / "ep0001.npz", {**ep, "label": np.int32(1)})

        ds = LabeledTrajectoryDataset(tmpdir, window_len=window_len, seed=42)
        assert len(ds) > 0

        sample = ds[0]
        assert hasattr(ds, "_last_window_start")
        assert sample["image"].shape == (window_len, 16, 16, 3), sample["image"].shape
        assert sample["action"].shape == (window_len, 7)
        assert "is_attack" in sample
        assert sample["image"].max() <= 1.0 and sample["image"].min() >= 0.0

        # Collate
        batch = collate_episodes([ds[i] for i in range(4)])
        assert batch["image"].shape == (4, window_len, 16, 16, 3)

    print("dataset: all smoke checks passed.")
