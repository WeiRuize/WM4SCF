"""
reference_bank.py
=================
Success Reference Bank for the Safety Critic fast layer.

Purpose:
  Stores projected VLA hidden states from successful rollouts.
  At inference time, provides ρ_t(S) — the per-step reference vector
  used to compute the residual δ_t = h_t_proj − ρ_t(S).

Design:
  - Maintains a fixed-capacity FIFO queue of projected hidden vectors.
  - At each step t, ρ_t(S) = mean of all stored vectors within a step-
    aligned window (or global mean if step-alignment is disabled).
  - Thread-safe for single-process online inference.

Usage:
  bank = ReferenceBank(capacity=10000, proj_dim=512)

  # During reference collection (clean rollouts):
  bank.add(h_proj)          # h_proj: (proj_dim,) or (B, proj_dim)

  # At inference:
  rho = bank.query(step=t)  # returns (proj_dim,) reference vector
"""

from __future__ import annotations
import collections
from typing import Optional

import numpy as np
import torch


class ReferenceBank:
    """
    FIFO buffer of projected VLA hidden states from successful rollouts.

    Args:
        capacity:    Maximum number of stored vectors.
        proj_dim:    Dimension of each stored vector (must match critic proj_dim).
        step_bins:   If > 0, compute per-step-bin mean for temporal alignment.
                     step t is mapped to bin t % step_bins.
                     If 0, use global mean regardless of step.
        device:      Device to return tensors on.
    """

    def __init__(
        self,
        capacity: int = 10000,
        proj_dim: int = 512,
        step_bins: int = 0,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.proj_dim = proj_dim
        self.step_bins = step_bins
        self.device = device

        # Global FIFO buffer
        self._buffer: collections.deque = collections.deque(maxlen=capacity)

        # Per-bin buffers (optional)
        self._bin_buffers: Optional[dict] = None
        if step_bins > 0:
            bin_cap = max(1, capacity // step_bins)
            self._bin_buffers = {
                b: collections.deque(maxlen=bin_cap)
                for b in range(step_bins)
            }

    # ── Adding vectors ─────────────────────────────────────────────────────

    def add(
        self,
        h_proj: torch.Tensor,
        step: Optional[int] = None,
    ) -> None:
        """
        Add projected hidden state(s) to the bank.

        Args:
            h_proj: (proj_dim,) or (B, proj_dim)  float32
            step:   rollout step index for bin-aware storage
        """
        if h_proj.ndim == 1:
            h_proj = h_proj.unsqueeze(0)
        vecs = h_proj.detach().float().cpu().numpy()   # (B, proj_dim)
        for v in vecs:
            self._buffer.append(v)
            if self._bin_buffers is not None and step is not None:
                b = step % self.step_bins
                self._bin_buffers[b].append(v)

    def add_episode(
        self,
        h_proj_seq: torch.Tensor,
    ) -> None:
        """
        Add all steps of a projected hidden-state sequence.

        Args:
            h_proj_seq: (T, proj_dim) float32
        """
        vecs = h_proj_seq.detach().float().cpu().numpy()
        for t, v in enumerate(vecs):
            self.add(torch.from_numpy(v), step=t)

    # ── Querying ───────────────────────────────────────────────────────────

    def query(self, step: Optional[int] = None) -> torch.Tensor:
        """
        Compute the reference vector ρ_t(S) for step t.

        If step_bins > 0 and step is given, returns the mean over the
        corresponding per-bin buffer.  Falls back to the global mean if
        the bin is empty.

        Args:
            step: current rollout step (0-indexed)

        Returns:
            rho: (proj_dim,) float32 on self.device
        """
        if self._bin_buffers is not None and step is not None:
            b = step % self.step_bins
            buf = self._bin_buffers[b]
            if len(buf) > 0:
                arr = np.stack(list(buf), axis=0)
                return torch.from_numpy(arr.mean(axis=0)).float().to(self.device)

        # Global mean fallback
        return self.global_mean()

    def query_batch(self, steps: torch.Tensor) -> torch.Tensor:
        """
        Query references for a batch of step indices.

        Args:
            steps: (B,) int64 step indices

        Returns:
            rho: (B, proj_dim) float32
        """
        refs = [self.query(int(s.item())) for s in steps]
        return torch.stack(refs, dim=0)

    def global_mean(self) -> torch.Tensor:
        """Return mean over the entire buffer."""
        if len(self._buffer) == 0:
            return torch.zeros(self.proj_dim, device=self.device)
        arr = np.stack(list(self._buffer), axis=0)
        return torch.from_numpy(arr.mean(axis=0)).float().to(self.device)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save bank contents as .npz."""
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(list(self._buffer), axis=0) if self._buffer else np.empty((0, self.proj_dim))
        meta = {
            "vectors": arr,
            "capacity": np.int32(self.capacity),
            "proj_dim": np.int32(self.proj_dim),
            "step_bins": np.int32(self.step_bins),
        }
        np.savez_compressed(path, **meta)
        print(f"[ReferenceBank] Saved {len(self._buffer)} vectors to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ReferenceBank":
        """Load bank from .npz."""
        data = np.load(path)
        bank = cls(
            capacity=int(data["capacity"]),
            proj_dim=int(data["proj_dim"]),
            step_bins=int(data["step_bins"]),
            device=device,
        )
        for v in data["vectors"]:
            bank._buffer.append(v)
        print(f"[ReferenceBank] Loaded {len(bank._buffer)} vectors from {path}")
        return bank

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReferenceBank(capacity={self.capacity}, proj_dim={self.proj_dim}, "
            f"stored={len(self._buffer)}, step_bins={self.step_bins})"
        )


if __name__ == "__main__":
    import tempfile, os

    proj_dim = 16
    bank = ReferenceBank(capacity=100, proj_dim=proj_dim, step_bins=10)
    assert len(bank) == 0

    # Add single vectors
    for t in range(30):
        v = torch.randn(proj_dim)
        bank.add(v, step=t)
    assert len(bank) == 30

    # Add batch
    batch = torch.randn(5, proj_dim)
    bank.add(batch, step=5)
    assert len(bank) == 35

    # Query global mean
    rho = bank.global_mean()
    assert rho.shape == (proj_dim,)

    # Query per-step
    rho_t = bank.query(step=3)
    assert rho_t.shape == (proj_dim,)

    # Batch query
    steps = torch.arange(5)
    rho_batch = bank.query_batch(steps)
    assert rho_batch.shape == (5, proj_dim)

    # Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bank.npz")
        bank.save(path)
        bank2 = ReferenceBank.load(path)
        assert len(bank2) == len(bank)
        diff = (bank2.global_mean() - bank.global_mean()).abs().max().item()
        assert diff < 1e-5, diff

    print("reference_bank: all smoke checks passed.")
