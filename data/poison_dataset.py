"""
poison_dataset.py
=================
Mixed clean + GOBA-poisoned LIBERO loader for Stage-2 Safety Critic
training with self-supervised intrinsic cost (see train/train_critic.py).

Design notes:
  - No per-demo "is_backdoor" labels are required.  The training signal is
    the frozen-RSSM one-step KL surprise, computed on-the-fly; poison
    files simply provide *harder* rollouts that produce larger surprise.
  - For balanced exposure to clean vs. poison dynamics during training we
    still need to know which file came from which root — this is derived
    from the directory, NOT from inspecting the HDF5.
  - HDF5 layout matches the authoritative LIBERO schema documented in
    data/libero_dataset.py (`obs/agentview_rgb`, `obs/joint_states`,
    `actions`).  Both clean and poison suites share this layout.
"""

from __future__ import annotations
import pathlib
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# Source tag: 0 = clean demonstration, 1 = poison demonstration.
# Used for balanced sampling and for logging aggregate statistics.
_CLEAN, _POISON = 0, 1


class MixedLiberoDataset(Dataset):
    """
    Mixed clean + poisoned window dataset.

    Each item is a contiguous `window_len`-step window from a randomly
    chosen demo.  Windows are sampled uniformly across demos; call
    `sample_weights()` if you want balanced clean/poison exposure via a
    WeightedRandomSampler.
    """

    def __init__(
        self,
        clean_root: str | pathlib.Path | None,
        poison_root: str | pathlib.Path | None,
        window_len: int = 64,
        image_size: int = 128,
        seed: int = 0,
    ):
        self.window_len = window_len
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

        # (hdf5_path, demo_key, T, source_tag)
        self.items: List[Tuple[pathlib.Path, str, int, int]] = []

        if clean_root is not None:
            self._index_root(pathlib.Path(clean_root), tag=_CLEAN)
        if poison_root is not None:
            self._index_root(pathlib.Path(poison_root), tag=_POISON)

        if not self.items:
            raise RuntimeError(
                f"[MixedLiberoDataset] no demos found under "
                f"clean={clean_root} poison={poison_root}"
            )
        n_cl = sum(1 for x in self.items if x[3] == _CLEAN)
        n_po = sum(1 for x in self.items if x[3] == _POISON)
        print(f"[MixedLiberoDataset] indexed {len(self.items)} demos "
              f"(clean={n_cl}, poison={n_po})")

    def _index_root(self, root: pathlib.Path, tag: int) -> None:
        if not root.exists():
            print(f"[MixedLiberoDataset] root not found, skipping: {root}")
            return
        for fp in sorted(root.rglob("*.hdf5")):
            try:
                with h5py.File(fp, "r") as f:
                    if "data" not in f:
                        continue
                    for key in f["data"].keys():
                        T = f[f"data/{key}/actions"].shape[0]
                        if T >= 2:
                            self.items.append((fp, key, T, tag))
            except OSError as e:
                print(f"[MixedLiberoDataset] skip unreadable {fp}: {e}")

    # ── Sampler helpers ──────────────────────────────────────────────────

    def sample_weights(self) -> np.ndarray:
        """Return per-item weights so clean and poison get equal total mass."""
        tags = np.array([x[3] for x in self.items], dtype=np.int64)
        n_po = int((tags == _POISON).sum())
        n_cl = int((tags == _CLEAN).sum())
        if n_cl == 0 or n_po == 0:
            return np.ones(len(tags), dtype=np.float32)
        w = np.where(tags == _POISON, 0.5 / n_po, 0.5 / n_cl)
        return w.astype(np.float32)

    # ── Pytorch interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        fp, key, T, tag = self.items[idx]
        L = min(self.window_len, T)
        start = self.rng.randint(0, T - L + 1) if T > L else 0

        with h5py.File(fp, "r") as f:
            grp = f[f"data/{key}"]
            img = grp["obs/agentview_rgb"][start:start + L].astype(np.uint8)
            proprio = grp["obs/joint_states"][start:start + L].astype(np.float32)
            action = grp["actions"][start:start + L].astype(np.float32)

        # Resize images if their native resolution differs from image_size.
        if img.shape[1] != self.image_size:
            t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
            t = torch.nn.functional.interpolate(
                t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            img = t.permute(0, 2, 3, 1).byte().numpy()

        # Right-pad to fixed window_len (with the terminal frame / zero action).
        if L < self.window_len:
            pad = self.window_len - L
            img = np.concatenate([img, np.repeat(img[-1:], pad, axis=0)], axis=0)
            proprio = np.concatenate(
                [proprio, np.repeat(proprio[-1:], pad, axis=0)], axis=0
            )
            action = np.concatenate(
                [action, np.zeros((pad, action.shape[1]), dtype=np.float32)], axis=0
            )

        is_first = np.zeros(self.window_len, dtype=np.float32)
        is_first[0] = 1.0

        return {
            "image": torch.from_numpy(img).float() / 255.0,
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
            "is_first": torch.from_numpy(is_first),
            "source": torch.tensor(int(tag), dtype=torch.long),
        }


def make_mixed_loader(
    clean_root: str | pathlib.Path | None,
    poison_root: str | pathlib.Path | None,
    window_len: int = 64,
    image_size: int = 128,
    batch_size: int = 16,
    num_workers: int = 4,
    balanced: bool = True,
    seed: int = 0,
) -> DataLoader:
    ds = MixedLiberoDataset(
        clean_root=clean_root,
        poison_root=poison_root,
        window_len=window_len,
        image_size=image_size,
        seed=seed,
    )
    if balanced:
        weights = torch.from_numpy(ds.sample_weights()).double()
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


# Legacy aliases — some scripts still import the old names.
PoisonDataset = MixedLiberoDataset
make_poison_loader = make_mixed_loader


if __name__ == "__main__":
    import sys
    clean = sys.argv[1] if len(sys.argv) > 1 else "no_noops_datasets"
    poison = sys.argv[2] if len(sys.argv) > 2 else "Poisoned_Dataset"
    try:
        ds = MixedLiberoDataset(clean, poison, window_len=16, image_size=64)
        item = ds[0]
        assert item["image"].shape == (16, 64, 64, 3)
        assert item["action"].shape == (16, 7)
        assert item["proprio"].shape == (16, 7)
        print("poison_dataset: smoke ok")
    except RuntimeError as e:
        print(f"poison_dataset: skipped live test ({e})")
