"""
openvla_wrapper.py
==================
Black-box wrapper around OpenVLA for SafeVLA inference.

Source interfaces reused from:
    openvla-main/experiments/robot/openvla_utils.py
        get_vla(), get_processor(), get_vla_action()
    openvla-main/prismatic/models/vlas/openvla.py
        OpenVLA.predict_action()

Contract:
  - Do NOT modify any VLA weights or internals.
  - Call only official inference interfaces.
  - Expose VLA hidden states via output_hidden_states=True at generate() time.

Hidden-state extraction:
  LLaMA-2 7B last-layer hidden state: outputs.hidden_states[-1][:, -1, :]
  Shape: (1, 4096) per inference step.
"""

from __future__ import annotations
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── Make openvla-main importable ──────────────────────────────────────────
_OPENVLA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "openvla-main")
if _OPENVLA_ROOT not in sys.path:
    sys.path.insert(0, _OPENVLA_ROOT)

try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    _OPENVLA_AVAILABLE = True
except ImportError:
    _OPENVLA_AVAILABLE = False
    print(
        "[OpenVLAWrapper] WARNING: openvla-main dependencies not installed. "
        "Call .load_model() will raise at runtime."
    )


OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _build_prompt(base_vla_name: str, task_label: str) -> str:
    if "openvla-v01" in base_vla_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take "
            f"to {task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


class OpenVLAWrapper:
    """
    Thin, black-box wrapper around OpenVLA-7B.

    Provides:
      predict(obs, task_label)              → action (7,)
      predict_with_hidden(obs, task_label)  → (action (7,), hidden (4096,))

    Args:
        pretrained_checkpoint: HF Hub ID or local path, e.g. "openvla/openvla-7b"
        unnorm_key:            Dataset key for action un-normalization
        device:                torch device string
        load_in_8bit:          Use bitsandbytes 8-bit quantization
        load_in_4bit:          Use bitsandbytes 4-bit quantization
        center_crop:           Apply center crop pre-processing (as in LIBERO eval)
    """

    def __init__(
        self,
        pretrained_checkpoint: str = "openvla/openvla-7b",
        unnorm_key: Optional[str] = None,
        device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        center_crop: bool = False,
    ):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.unnorm_key = unnorm_key
        self.device = torch.device(device)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.center_crop = center_crop

        self.vla = None
        self.processor = None
        self._base_name = os.path.basename(pretrained_checkpoint)

    # ── Model loading ──────────────────────────────────────────────────────

    def load_model(self) -> "OpenVLAWrapper":
        """
        Load VLA model and processor from checkpoint.
        Call this before any inference.  Server-side only.
        """
        if not _OPENVLA_AVAILABLE:
            raise RuntimeError(
                "openvla-main dependencies are not installed. "
                "Run `pip install -e openvla-main` on the server."
            )

        print(f"[OpenVLAWrapper] Loading {self.pretrained_checkpoint} ...")

        # Register custom HF model classes
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.pretrained_checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if not self.load_in_8bit and not self.load_in_4bit:
            self.vla = self.vla.to(self.device)
        self.vla.eval()

        # Load action normalization statistics if available locally
        stats_path = os.path.join(self.pretrained_checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                self.vla.norm_stats = json.load(f)

        self.processor = AutoProcessor.from_pretrained(
            self.pretrained_checkpoint, trust_remote_code=True
        )

        print("[OpenVLAWrapper] Model loaded.")
        return self

    # ── Observation pre-processing ─────────────────────────────────────────

    def _preprocess_image(self, image_np: np.ndarray) -> Image.Image:
        """Convert uint8 (H,W,3) numpy array to PIL Image."""
        img = Image.fromarray(image_np).convert("RGB")
        if self.center_crop:
            img = self._center_crop_pil(img)
        return img

    @staticmethod
    def _center_crop_pil(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
        """
        Center crop to area `crop_scale` × original, then resize back.
        Replicates the dlimp RLDS crop logic from openvla_utils.py.
        """
        try:
            import tensorflow as tf
        except ImportError:
            return image

        import numpy as np
        arr = np.array(image).astype(np.float32) / 255.0
        t = tf.constant(arr)[None]  # (1, H, W, 3)
        s = tf.sqrt(float(crop_scale))
        H, W = arr.shape[:2]
        off_h = (1.0 - s) / 2.0
        off_w = (1.0 - s) / 2.0
        boxes = tf.constant([[off_h, off_w, off_h + s, off_w + s]])
        cropped = tf.image.crop_and_resize(t, boxes, [0], (H, W))[0].numpy()
        cropped = np.clip(cropped * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(cropped)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.inference_mode()
    def predict(
        self,
        obs: dict,           # {"full_image": np.ndarray (H,W,3) uint8}
        task_label: str,
    ) -> np.ndarray:
        """
        Generate action without returning hidden states.

        Returns:
            action: (7,) float32 un-normalized action
        """
        action, _ = self._generate(obs, task_label, return_hidden=False)
        return action

    @torch.inference_mode()
    def predict_with_hidden(
        self,
        obs: dict,
        task_label: str,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Generate action AND extract last-layer hidden state.

        Returns:
            action:  (7,) float32 un-normalized action
            h_t:     (4096,) float32 last-layer hidden state (on CPU)
        """
        return self._generate(obs, task_label, return_hidden=True)

    def _generate(
        self,
        obs: dict,
        task_label: str,
        return_hidden: bool,
    ) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
        assert self.vla is not None, "Call load_model() first."

        image = self._preprocess_image(obs["full_image"])
        prompt = _build_prompt(self._base_name, task_label)

        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        action_dim = self.vla.get_action_dim(self.unnorm_key)

        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=action_dim,
            do_sample=False,
        )
        if return_hidden:
            generate_kwargs["output_hidden_states"] = True
            generate_kwargs["return_dict_in_generate"] = True

        outputs = self.vla.generate(**generate_kwargs)

        # Decode action tokens → continuous action
        if return_hidden:
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs

        predicted_ids = generated_ids[0, -action_dim:]
        norm_actions = self.vla.action_tokenizer.decode_token_ids_to_actions(
            predicted_ids.cpu().numpy()
        )

        # Un-normalize
        stats = self.vla.get_action_stats(self.unnorm_key)
        mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
        high, low = np.array(stats["q99"]), np.array(stats["q01"])
        action = np.where(mask, 0.5 * (norm_actions + 1) * (high - low) + low, norm_actions)

        h_t = None
        if return_hidden:
            # hidden_states: tuple(len=num_generated_tokens) of
            #   tuple(len=num_layers) of (batch, seq_len, hidden_dim)
            # Take last generated token, last layer
            last_token_hidden = outputs.hidden_states[-1]   # tuple of layers
            h_t = last_token_hidden[-1][:, -1, :]           # (1, hidden_dim)
            h_t = h_t.squeeze(0).float().cpu()              # (hidden_dim,)

        return action.astype(np.float32), h_t

    # ── Batch inference for training data collection ───────────────────────

    @torch.inference_mode()
    def collect_hidden_states(
        self,
        episode_obs: list,      # list of np.ndarray (H,W,3) uint8
        task_label: str,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Run VLA over a list of observations; collect all actions and hidden states.
        Used offline for building the training dataset for the critic.

        Returns:
            actions: (T, 7)     float32
            hiddens: (T, 4096)  float32
        """
        actions, hiddens = [], []
        for obs_frame in episode_obs:
            obs = {"full_image": obs_frame}
            action, h_t = self.predict_with_hidden(obs, task_label)
            actions.append(action)
            hiddens.append(h_t)
        return np.stack(actions, axis=0), torch.stack(hiddens, dim=0)


if __name__ == "__main__":
    # Smoke test: verify import path resolution without loading a real model.
    print(f"[OpenVLAWrapper] openvla-main root resolved to: {_OPENVLA_ROOT}")
    assert os.path.isdir(_OPENVLA_ROOT), (
        f"openvla-main not found at {_OPENVLA_ROOT}. "
        "Ensure openvla-main/ is a sibling of the project root."
    )

    wrapper = OpenVLAWrapper(pretrained_checkpoint="openvla/openvla-7b")
    print(f"[OpenVLAWrapper] Wrapper created: {wrapper}")
    print("openvla_wrapper: import and path smoke checks passed.")
    print("NOTE: call load_model() and predict_with_hidden() on the server with a GPU.")
