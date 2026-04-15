"""
encode_trajectories.py
======================
Pre-extract VLA hidden states for all episodes in a dataset directory.

This must be run on the server (GPU required) BEFORE training the critic.
It reads each .npz episode, runs OpenVLA frame-by-frame, and writes the
hidden states back as a new "vla_hidden" key in the same .npz file.

Run on server:
    cd <project_root>
    python train/encode_trajectories.py \\
        --data_dir   data/labeled_trajectories \\
        --task_label "pick up the block and place it in the bowl" \\
        --checkpoint openvla/openvla-7b \\
        --unnorm_key libero_spatial \\
        --device     cuda:0 \\
        --overwrite

The resulting .npz files gain an extra array:
    vla_hidden: (T, 4096)  float32
"""

import argparse
import pathlib
import sys

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from data.dataset import load_episode, save_episode
from vla.openvla_wrapper import OpenVLAWrapper


def encode_episode(
    wrapper: OpenVLAWrapper,
    episode: dict,
    task_label: str,
) -> np.ndarray:
    """
    Run VLA frame-by-frame and collect hidden states.

    Args:
        episode:    episode dict with "image" key (T, H, W, 3) uint8
        task_label: natural-language task description

    Returns:
        hiddens: (T, 4096) float32
    """
    T = episode["image"].shape[0]
    hiddens = []
    for t in range(T):
        obs = {"full_image": episode["image"][t]}
        _, h_t = wrapper.predict_with_hidden(obs, task_label)  # (4096,)
        hiddens.append(h_t.numpy())
    return np.stack(hiddens, axis=0).astype(np.float32)   # (T, 4096)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,  help="Directory of .npz episodes")
    parser.add_argument("--task_label", required=True,  help="Task instruction string")
    parser.add_argument("--checkpoint", default="openvla/openvla-7b")
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--overwrite",  action="store_true",
                        help="Re-encode even if 'vla_hidden' already present")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    episodes = sorted(data_dir.glob("*.npz"))
    print(f"[encode_trajectories] Found {len(episodes)} episodes in {data_dir}")

    wrapper = OpenVLAWrapper(
        pretrained_checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        device=args.device,
    ).load_model()

    skipped, encoded = 0, 0
    for i, ep_path in enumerate(episodes):
        ep = load_episode(ep_path)

        if "vla_hidden" in ep and not args.overwrite:
            skipped += 1
            continue

        try:
            hiddens = encode_episode(wrapper, ep, args.task_label)
            ep["vla_hidden"] = hiddens
            save_episode(ep_path, ep)
            encoded += 1
        except Exception as e:
            print(f"  [ERROR] {ep_path.name}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(episodes)}] encoded={encoded}, skipped={skipped}")

    print(f"[encode_trajectories] Done. Encoded={encoded}, Skipped={skipped}.")


if __name__ == "__main__":
    main()
