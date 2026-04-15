"""
encoder.py
==========
Robot observation encoder for SafeVLA world-model training.
Adapted from dreamerv3-torch-main/networks.py  ConvEncoder + MLP.

Handles two input streams:
  1. RGB image  (H, W, C)  → CNN branch
  2. Proprioception vector → MLP branch

Both branches are concatenated into a flat embedding that feeds the RSSM.
"""

import math
import numpy as np
import torch
from torch import nn

from models.dreamer_utils import (
    Conv2dSamePad,
    ImgChLayerNorm,
    weight_init,
)


class ImageEncoder(nn.Module):
    """
    Strided-CNN image encoder.
    Adapted from dreamerv3-torch networks.ConvEncoder.

    Input:  (B, T, H, W, C) – float32 in [0, 1]  or  (B, H, W, C)
    Output: (B, T, outdim)                         or  (B, outdim)
    """

    def __init__(
        self,
        input_shape,          # (H, W, C)
        cnn_depth: int = 32,
        act: str = "SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        h, w, in_ch = input_shape
        stages = int(np.log2(h) - np.log2(minres))
        in_dim, out_dim = in_ch, cnn_depth
        layers = []
        for _ in range(stages):
            layers.append(
                Conv2dSamePad(in_dim, out_dim, kernel_size=kernel_size, stride=2, bias=False)
            )
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act_cls())
            in_dim = out_dim
            out_dim *= 2
            h, w = h // 2, w // 2
        self.outdim = (out_dim // 2) * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (..., H, W, C) float32 in [0, 1]
        Returns:
            (..., outdim)
        """
        lead_shape = obs.shape[:-3]
        obs = obs - 0.5
        x = obs.reshape((-1,) + obs.shape[-3:])    # (N, H, W, C)
        x = x.permute(0, 3, 1, 2)                  # (N, C, H, W)
        x = self.layers(x)
        x = x.reshape(x.shape[0], -1)              # (N, outdim)
        return x.reshape(lead_shape + (x.shape[-1],))


class ProprioEncoder(nn.Module):
    """
    MLP encoder for proprioception vectors.

    Input:  (..., proprio_dim)
    Output: (..., units)
    """

    def __init__(
        self,
        proprio_dim: int = 7,
        layers: int = 3,
        units: int = 256,
        act: str = "SiLU",
        norm: bool = True,
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        net = []
        in_dim = proprio_dim
        for _ in range(layers):
            net.append(nn.Linear(in_dim, units, bias=False))
            if norm:
                net.append(nn.LayerNorm(units, eps=1e-3))
            net.append(act_cls())
            in_dim = units
        self.net = nn.Sequential(*net)
        self.net.apply(weight_init)
        self.outdim = units

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class RobotObsEncoder(nn.Module):
    """
    Combined image + proprioception encoder for SafeVLA RSSM.

    Concatenates image-CNN embedding and proprio-MLP embedding.
    Output shape: (..., embed_dim)  where embed_dim = image_outdim + proprio_units.
    """

    def __init__(
        self,
        image_shape=(224, 224, 3),
        cnn_depth: int = 32,
        kernel_size: int = 4,
        minres: int = 4,
        act: str = "SiLU",
        norm: bool = True,
        proprio_dim: int = 7,
        proprio_layers: int = 3,
        proprio_units: int = 256,
    ):
        super().__init__()
        self.image_enc = ImageEncoder(
            input_shape=image_shape,
            cnn_depth=cnn_depth,
            act=act,
            norm=norm,
            kernel_size=kernel_size,
            minres=minres,
        )
        self.proprio_enc = ProprioEncoder(
            proprio_dim=proprio_dim,
            layers=proprio_layers,
            units=proprio_units,
            act=act,
            norm=norm,
        )
        self.embed_dim = self.image_enc.outdim + self.proprio_enc.outdim

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: dict with keys:
                 "image":  (..., H, W, C) float32 in [0, 1]
                 "proprio": (..., proprio_dim) float32

        Returns:
            embed: (..., embed_dim)
        """
        img_emb = self.image_enc(obs["image"])
        prop_emb = self.proprio_enc(obs["proprio"])
        return torch.cat([img_emb, prop_emb], dim=-1)

    @classmethod
    def from_config(cls, cfg: dict) -> "RobotObsEncoder":
        enc_cfg = cfg["encoder"]
        return cls(
            image_shape=tuple(enc_cfg["image_shape"]),
            cnn_depth=enc_cfg["cnn_depth"],
            kernel_size=enc_cfg["kernel_size"],
            minres=enc_cfg["minres"],
            act=enc_cfg["act"],
            norm=enc_cfg["norm"],
            proprio_dim=enc_cfg["proprio_dim"],
            proprio_layers=enc_cfg["proprio_layers"],
            proprio_units=enc_cfg["proprio_units"],
        )


if __name__ == "__main__":
    B, T = 2, 4
    H, W, C = 64, 64, 3    # use small image for CPU smoke test
    proprio_dim = 7

    enc = RobotObsEncoder(
        image_shape=(H, W, C),
        cnn_depth=8,
        minres=4,
        proprio_dim=proprio_dim,
        proprio_layers=2,
        proprio_units=16,
    )

    obs = {
        "image": torch.rand(B, T, H, W, C),
        "proprio": torch.rand(B, T, proprio_dim),
    }
    embed = enc(obs)
    assert embed.shape[0] == B and embed.shape[1] == T
    assert embed.shape[-1] == enc.embed_dim

    print(f"encoder: embed_dim={enc.embed_dim}, output={embed.shape} — smoke checks passed.")
