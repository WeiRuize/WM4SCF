"""
decoder.py
==========
Observation decoder for SafeVLA RSSM training.
Adapted from dreamerv3-torch-main/networks.py  ConvDecoder + MLP.

Used only during world-model training (train_rssm.py).
Not loaded during inference by the Safety Guardian.

Heads:
  - Image decoder (ConvTranspose2d stack) → reconstructed RGB
  - Proprio decoder (MLP) → reconstructed proprioception
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.dreamer_utils import (
    ImgChLayerNorm,
    weight_init,
    uniform_weight_init,
    MSEDist,
    SymlogDist,
)


class ImageDecoder(nn.Module):
    """
    Transposed-CNN decoder: RSSM feature → RGB image.
    Adapted from dreamerv3-torch networks.ConvDecoder.

    Input:  (..., feat_size)
    Output: (..., H, W, C)  in [0, 1]
    """

    def __init__(
        self,
        feat_size: int,
        output_shape=(224, 224, 3),
        cnn_depth: int = 32,
        act: str = "SiLU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False,
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        self._output_shape = output_shape   # (H, W, C)
        self._cnn_sigmoid = cnn_sigmoid
        H, W, out_ch = output_shape

        layer_num = int(np.log2(H) - np.log2(minres))
        self._minres = minres

        # Linear projection: feat → (minres * minres * first_conv_channels)
        # first_conv_channels = cnn_depth * 2^(layer_num-1)
        first_ch = cnn_depth * (2 ** (layer_num - 1))
        out_ch_start = minres ** 2 * first_ch
        self._embed_size = out_ch_start
        self._first_ch = first_ch

        self._linear = nn.Linear(feat_size, out_ch_start)
        self._linear.apply(uniform_weight_init(outscale))

        # Build a clean channel schedule: [base*2^(n-1), base*2^(n-2), ..., base, out_ch]
        # where base = cnn_depth, n = layer_num.
        # Each layer halves the channel count; the final layer outputs out_ch (RGB).
        base = cnn_depth
        ch_schedule = [base * (2 ** (layer_num - 1 - i)) for i in range(layer_num)]
        ch_schedule.append(out_ch)   # final output channels

        pad_h, outpad_h = self._pad_pair(kernel_size)
        layers = []
        for i in range(layer_num):
            in_c  = ch_schedule[i]
            out_c = ch_schedule[i + 1]
            is_last = (i == layer_num - 1)
            layers.append(
                nn.ConvTranspose2d(
                    in_c, out_c, kernel_size, stride=2,
                    padding=(pad_h, pad_h),
                    output_padding=(outpad_h, outpad_h),
                    bias=is_last,
                )
            )
            if norm and not is_last:
                layers.append(ImgChLayerNorm(out_c))
            if not is_last:
                layers.append(act_cls())

        # Apply initialisation: uniform outscale to the last ConvTranspose2d,
        # standard weight_init to everything else.
        conv_layers = [m for m in layers if isinstance(m, nn.ConvTranspose2d)]
        for m in layers:
            if m is conv_layers[-1]:
                m.apply(uniform_weight_init(outscale))
            else:
                m.apply(weight_init)
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _pad_pair(k, s=2, d=1):
        val = d * (k - 1) - s + 1
        pad = math.ceil(val / 2)
        outpad = pad * 2 - val
        return pad, outpad

    @staticmethod
    def _calc_same_pad(k, s=2, d=1):
        val = d * (k - 1) - s + 1
        return math.ceil(val / 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (..., feat_size)
        Returns:
            mean:  (..., H, W, C) in [0, 1]
        """
        lead = features.shape[:-1]
        x = self._linear(features)
        # Reshape to (N, minres, minres, first_ch) then permute to NCHW
        x = x.reshape(-1, self._minres, self._minres, self._first_ch)
        x = x.permute(0, 3, 1, 2)                 # (N, first_ch, minres, minres)
        x = self.layers(x)                         # (N, out_ch, H, W)
        H, W, C = self._output_shape
        # Permute back to (..., H, W, C)
        mean = x.reshape(lead + (C, H, W))         # (..., C, H, W)
        mean = mean.permute(*range(len(lead)), -2, -1, -3)  # (..., H, W, C)
        if self._cnn_sigmoid:
            mean = torch.sigmoid(mean)
        else:
            mean = mean + 0.5
        return mean


class ProprioDecoder(nn.Module):
    """
    MLP decoder: RSSM feature → proprioception reconstruction.

    Input:  (..., feat_size)
    Output: distribution over (..., proprio_dim)
    """

    def __init__(
        self,
        feat_size: int,
        proprio_dim: int = 7,
        layers: int = 3,
        units: int = 256,
        act: str = "SiLU",
        norm: bool = True,
        dist: str = "symlog_mse",
        outscale: float = 1.0,
    ):
        super().__init__()
        act_cls = getattr(nn, act)
        net = []
        in_dim = feat_size
        for _ in range(layers):
            net.append(nn.Linear(in_dim, units, bias=False))
            if norm:
                net.append(nn.LayerNorm(units, eps=1e-3))
            net.append(act_cls())
            in_dim = units
        self.net = nn.Sequential(*net)
        self.net.apply(weight_init)
        self.mean_layer = nn.Linear(units, proprio_dim)
        self.mean_layer.apply(uniform_weight_init(outscale))
        self._dist = dist

    def forward(self, features: torch.Tensor):
        """Returns a distribution object with .log_prob(target)."""
        out = self.net(features)
        mean = self.mean_layer(out)
        if self._dist == "symlog_mse":
            return SymlogDist(mean)
        elif self._dist == "mse":
            return MSEDist(mean)
        raise NotImplementedError(self._dist)


class RobotObsDecoder(nn.Module):
    """
    Combined image + proprio decoder for RSSM training.

    Returns a dict of distribution objects:
        {"image": ImageDist, "proprio": ProprioDistribution}
    """

    def __init__(
        self,
        feat_size: int,
        image_shape=(224, 224, 3),
        cnn_depth: int = 32,
        kernel_size: int = 4,
        minres: int = 4,
        act: str = "SiLU",
        norm: bool = True,
        cnn_sigmoid: bool = False,
        image_dist: str = "mse",
        proprio_dim: int = 7,
        mlp_layers: int = 3,
        mlp_units: int = 256,
        vector_dist: str = "symlog_mse",
        outscale: float = 1.0,
    ):
        super().__init__()
        self.image_dec = ImageDecoder(
            feat_size=feat_size,
            output_shape=image_shape,
            cnn_depth=cnn_depth,
            act=act,
            norm=norm,
            kernel_size=kernel_size,
            minres=minres,
            outscale=outscale,
            cnn_sigmoid=cnn_sigmoid,
        )
        self.proprio_dec = ProprioDecoder(
            feat_size=feat_size,
            proprio_dim=proprio_dim,
            layers=mlp_layers,
            units=mlp_units,
            act=act,
            norm=norm,
            dist=vector_dist,
            outscale=outscale,
        )
        self._image_dist = image_dist

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (..., feat_size)
        Returns:
            {"image": dist, "proprio": dist}
        """
        img_mean = self.image_dec(features)
        if self._image_dist == "mse":
            img_dist = MSEDist(img_mean)
        else:
            raise NotImplementedError(self._image_dist)
        proprio_dist = self.proprio_dec(features)
        return {"image": img_dist, "proprio": proprio_dist}

    @classmethod
    def from_config(cls, cfg: dict, feat_size: int) -> "RobotObsDecoder":
        dec_cfg = cfg["decoder"]
        enc_cfg = cfg["encoder"]
        return cls(
            feat_size=feat_size,
            image_shape=tuple(enc_cfg["image_shape"]),
            cnn_depth=dec_cfg["cnn_depth"],
            kernel_size=dec_cfg["kernel_size"],
            minres=dec_cfg["minres"],
            act=dec_cfg["act"],
            norm=dec_cfg["norm"],
            cnn_sigmoid=dec_cfg["cnn_sigmoid"],
            image_dist=dec_cfg["image_dist"],
            proprio_dim=enc_cfg["proprio_dim"],
            mlp_layers=dec_cfg["mlp_layers"],
            mlp_units=dec_cfg["mlp_units"],
            vector_dist=dec_cfg["vector_dist"],
            outscale=dec_cfg["outscale"],
        )


if __name__ == "__main__":
    B, T = 2, 3
    H, W, C = 32, 32, 3     # small for CPU
    feat_size = 32
    proprio_dim = 7

    dec = RobotObsDecoder(
        feat_size=feat_size,
        image_shape=(H, W, C),
        cnn_depth=4,
        minres=4,
        act="SiLU",
        norm=True,
        proprio_dim=proprio_dim,
        mlp_layers=2,
        mlp_units=16,
    )

    feat = torch.randn(B, T, feat_size)
    dists = dec(feat)

    # Image reconstruction
    img_mean = dists["image"].mode()
    assert img_mean.shape == (B, T, H, W, C), img_mean.shape

    # Proprio log-prob
    target_proprio = torch.randn(B, T, proprio_dim)
    lp = dists["proprio"].log_prob(target_proprio)
    assert lp.shape == (B, T), lp.shape

    print(f"decoder: image={img_mean.shape}, proprio_lp={lp.shape} — smoke checks passed.")
