"""
vla/pi0_wrapper.py
==================
Black-box wrapper for a π₀ (action-chunk) VLA.  Only the `predict_chunk`
method is called by the Safety Guardian; every other attribute of the
underlying model stays hidden.

This is a thin stub that must be filled in once the project decides which
π₀ checkpoint family to use (e.g. Physical-Intelligence π₀ release).  The
interface is fixed so downstream eval code can import it unchanged.
"""

from __future__ import annotations
import torch


class Pi0Wrapper:
    def __init__(
        self,
        pretrained_checkpoint: str,
        chunk_horizon: int = 8,
        device: str = "cuda:0",
    ):
        self.checkpoint = pretrained_checkpoint
        self.chunk_horizon = chunk_horizon
        self.device = device
        # TODO: load the real π₀ model here once the checkpoint source is
        # selected.  The interface below is frozen.
        self._model = None

    @torch.no_grad()
    def predict_chunk(self, obs: dict, task_label: str) -> torch.Tensor:
        """
        Args:
            obs:        dict with keys "image" (H, W, 3) uint8 and
                        "proprio" (D,) float32
            task_label: natural-language instruction
        Returns:
            action_chunk: (chunk_horizon, 7) float32 on CPU
        """
        if self._model is None:
            raise NotImplementedError(
                "Pi0Wrapper is a stub; load a real π₀ checkpoint before use."
            )
        # The real call would look like:
        #   chunk = self._model.predict(obs, task_label, horizon=self.chunk_horizon)
        raise NotImplementedError
