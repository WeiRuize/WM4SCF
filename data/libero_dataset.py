"""
libero_dataset.py
=================
Clean LIBERO HDF5 dataset loader for RSSM world-model training (阶段一).

Only clean data is touched here — never load GOBA-poisoned episodes through
this class.  A separate loader (`poison_dataset.py`) handles the poisoned
split for Safety Critic training.

HDF5 layout (LIBERO convention)
    data/
      demo_0/
        actions           (T, 7)
        obs/agentview_rgb (T, 128, 128, 3) uint8
        obs/ee_states     (T, 7)           float32
      demo_1/
        ...

Each `__getitem__` samples a contiguous window of length `window_len` from a
randomly chosen demo inside a randomly chosen HDF5 file.  Images are
down-sampled to `(image_size, image_size)` to match the encoder input.
"""

from __future__ import annotations
import pathlib
import random
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LiberoDataset(Dataset):
    def __init__(
        self,
        root: str | pathlib.Path,
        window_len: int = 64,
        image_size: int = 128,
        suites: Tuple[str, ...] = (
            "libero_10_no_noops",
            "libero_goal_no_noops",
            "libero_object_no_noops",
            "libero_spatial_no_noops",
        ),
        seed: int = 0,
    ):
        self.root = pathlib.Path(root)
        self.window_len = window_len
        self.image_size = image_size
        self.rng = random.Random(seed)

        # Index of (hdf5_path, demo_key, demo_length)
        self.index: List[Tuple[pathlib.Path, str, int]] = []
        for suite in suites:
            suite_dir = self.root / suite
            if not suite_dir.exists():
                continue
            for fp in sorted(suite_dir.glob("*.hdf5")):
                with h5py.File(fp, "r") as f:
                    if "data" not in f:
                        continue
                    for key in f["data"].keys():
                        T = f[f"data/{key}/actions"].shape[0]
                        if T >= 2:
                            self.index.append((fp, key, T))
        if not self.index:
            raise RuntimeError(f"[LiberoDataset] No demos under {self.root}")
        print(f"[LiberoDataset] Indexed {len(self.index)} demos from {self.root}")

    def __len__(self) -> int:
        return len(self.index) * 10   # oversample; RSSM training uses steps, not epochs

    def _sample_window(self, fp: pathlib.Path, key: str, T: int) -> dict:
        L = min(self.window_len, T)
        start = self.rng.randint(0, T - L) if T > L else 0
        with h5py.File(fp, "r") as f:
            grp = f[f"data/{key}"]
            img = grp["obs/agentview_rgb"][start:start + L]
            proprio = grp["obs/ee_states"][start:start + L]
            action = grp["actions"][start:start + L]

        window = {
            "image": img.astype(np.uint8),
            "proprio": proprio.astype(np.float32),
            "action": action.astype(np.float32),
        }
        # Resize images if needed (bilinear via torch)
        if window["image"].shape[1] != self.image_size:
            t = torch.from_numpy(window["image"]).permute(0, 3, 1, 2).float()
            t = torch.nn.functional.interpolate(
                t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            window["image"] = t.permute(0, 2, 3, 1).byte().numpy()

        if L < self.window_len:
            pad = self.window_len - L
            window["image"] = np.concatenate(
                [window["image"], np.repeat(window["image"][-1:], pad, axis=0)], axis=0
            )
            window["proprio"] = np.concatenate(
                [window["proprio"], np.repeat(window["proprio"][-1:], pad, axis=0)], axis=0
            )
            window["action"] = np.concatenate(
                [window["action"], np.zeros((pad, window["action"].shape[1]), dtype=np.float32)],
                axis=0,
            )

        is_first = np.zeros(self.window_len, dtype=np.float32)
        is_first[0] = 1.0
        window["is_first"] = is_first
        return window

    def __getitem__(self, idx: int) -> dict:
        fp, key, T = self.index[idx % len(self.index)]
        w = self._sample_window(fp, key, T)
        return {
            "image": torch.from_numpy(w["image"]).float() / 255.0,
            "proprio": torch.from_numpy(w["proprio"]),
            "action": torch.from_numpy(w["action"]),
            "is_first": torch.from_numpy(w["is_first"]),
        }


def make_libero_loader(
    root: str | pathlib.Path,
    window_len: int = 64,
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    ds = LiberoDataset(root=root, window_len=window_len, image_size=image_size, **kwargs)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "no_noops_datasets"
    try:
        ds = LiberoDataset(root, window_len=16, image_size=64)
        sample = ds[0]
        assert sample["image"].shape == (16, 64, 64, 3)
        assert sample["action"].shape == (16, 7)
        print(f"libero_dataset: sample shapes OK, num_demos={len(ds.index)}")
    except RuntimeError as e:
        print(f"libero_dataset: skipped live test ({e})")
