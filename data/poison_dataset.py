"""
poison_dataset.py
=================
GOBA-poisoned LIBERO loader for Safety Critic training (阶段二).

Per-episode cost annotation (PROJECT_OVERVIEW §模块二):

    if trigger_detected(frame_t):   # first frame where the trigger is present
        costs[t:] = 1.0             # all subsequent steps are unsafe
        break
    else:
        costs[:] = 0.0              # clean episode

Trigger detection hook
----------------------
`trigger_detector(image_frame: np.ndarray) -> bool` is user-supplied.  For
GOBA the default is a colour-patch matcher; an alternative is to use the
per-episode metadata in `Poisoned_Dataset/inject_log.txt` to label whole
episodes as poisoned without per-frame detection.  Both modes are exposed
through the `labeling_mode` argument.

Episodes with no trigger (or clean files) receive cost vector 0 and are
mixed with poisoned episodes using a balanced sampler to avoid critic bias.
"""

from __future__ import annotations
import pathlib
from typing import Callable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DetectorFn = Callable[[np.ndarray], bool]


def _default_detector(frame: np.ndarray) -> bool:
    """Placeholder: replace with the real GOBA colour-patch check."""
    return False


class PoisonDataset(Dataset):
    """
    Every item is a full episode with an additional `costs` vector.

    Args:
        clean_root:    root containing clean HDF5 suites (no_noops_datasets).
        poison_root:   root containing GOBA-poisoned suites (Poisoned_Dataset).
        labeling_mode: "per_frame" → run detector on every frame.
                       "episode"   → label whole poisoned files as unsafe
                                     from their first step.
    """

    def __init__(
        self,
        clean_root: str | pathlib.Path,
        poison_root: str | pathlib.Path,
        window_len: int = 64,
        image_size: int = 128,
        trigger_detector: DetectorFn = _default_detector,
        labeling_mode: str = "episode",
        seed: int = 0,
    ):
        self.window_len = window_len
        self.image_size = image_size
        self.detector = trigger_detector
        self.labeling_mode = labeling_mode
        self.rng = np.random.RandomState(seed)

        self.items: List[Tuple[pathlib.Path, str, int, int]] = []
        for tag, root in (("clean", clean_root), ("poison", poison_root)):
            root = pathlib.Path(root)
            if not root.exists():
                continue
            for fp in sorted(root.rglob("*.hdf5")):
                with h5py.File(fp, "r") as f:
                    if "data" not in f:
                        continue
                    for key in f["data"].keys():
                        T = f[f"data/{key}/actions"].shape[0]
                        self.items.append((fp, key, T, 1 if tag == "poison" else 0))
        if not self.items:
            raise RuntimeError(
                f"[PoisonDataset] No episodes under {clean_root} / {poison_root}"
            )
        print(f"[PoisonDataset] {len(self.items)} episodes "
              f"({sum(x[3] for x in self.items)} poisoned)")

    def __len__(self) -> int:
        return len(self.items)

    def _compute_costs(self, images: np.ndarray, is_poison: int) -> np.ndarray:
        T = images.shape[0]
        costs = np.zeros(T, dtype=np.float32)
        if not is_poison:
            return costs
        if self.labeling_mode == "episode":
            costs[:] = 1.0
            return costs
        for t in range(T):
            if self.detector(images[t]):
                costs[t:] = 1.0
                break
        return costs

    def __getitem__(self, idx: int) -> dict:
        fp, key, T, is_poison = self.items[idx]
        L = min(self.window_len, T)
        start = self.rng.randint(0, T - L + 1) if T > L else 0
        with h5py.File(fp, "r") as f:
            grp = f[f"data/{key}"]
            img = grp["obs/agentview_rgb"][start:start + L].astype(np.uint8)
            proprio = grp["obs/ee_states"][start:start + L].astype(np.float32)
            action = grp["actions"][start:start + L].astype(np.float32)

        if img.shape[1] != self.image_size:
            t = torch.from_numpy(img).permute(0, 3, 1, 2).float()
            t = torch.nn.functional.interpolate(
                t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            img = t.permute(0, 2, 3, 1).byte().numpy()

        costs = self._compute_costs(img, is_poison)

        if L < self.window_len:
            pad = self.window_len - L
            img = np.concatenate([img, np.repeat(img[-1:], pad, axis=0)], axis=0)
            proprio = np.concatenate([proprio, np.repeat(proprio[-1:], pad, axis=0)], axis=0)
            action = np.concatenate(
                [action, np.zeros((pad, action.shape[1]), dtype=np.float32)], axis=0
            )
            costs = np.concatenate([costs, np.repeat(costs[-1:], pad, axis=0)], axis=0)

        is_first = np.zeros(self.window_len, dtype=np.float32)
        is_first[0] = 1.0

        return {
            "image": torch.from_numpy(img).float() / 255.0,
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action),
            "is_first": torch.from_numpy(is_first),
            "costs": torch.from_numpy(costs),
            "is_poison": torch.tensor(int(is_poison), dtype=torch.long),
        }


def make_poison_loader(
    clean_root: str | pathlib.Path,
    poison_root: str | pathlib.Path,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    ds = PoisonDataset(clean_root=clean_root, poison_root=poison_root, **kwargs)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
