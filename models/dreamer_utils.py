"""
dreamer_utils.py
================
Utilities extracted and adapted from:
    dreamerv3-torch-main/tools.py   (weight_init, static_scan, distributions, Optimizer)
    dreamerv3-torch-main/networks.py (GRUCell, Conv2dSamePad, ImgChLayerNorm)

Source: https://github.com/NM512/dreamerv3-torch
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd

# ── Weight initialisation ──────────────────────────────────────────────────

def weight_init(m):
    """Truncated-normal initialisation for Linear / Conv / LayerNorm."""
    if isinstance(m, (nn.Linear,)):
        in_num, out_num = m.in_features, m.out_features
        scale = 1.0 / ((in_num + out_num) / 2.0)
        std = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        scale = 1.0 / ((in_num + out_num) / 2.0)
        std = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2 * std, b=2 * std)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    """Uniform initialisation with a user-supplied output scale."""
    def f(m):
        if isinstance(m, (nn.Linear,)):
            in_num, out_num = m.in_features, m.out_features
            scale = given_scale / ((in_num + out_num) / 2.0)
            limit = math.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            scale = given_scale / ((in_num + out_num) / 2.0)
            limit = math.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    return f


# ── Sequential scan ────────────────────────────────────────────────────────

def static_scan(fn, inputs, start):
    """
    Apply fn recurrently over the first dimension of each input tensor.
    Identical to dreamerv3-torch tools.static_scan.

    Args:
        fn:     callable(prev_state, *step_inputs) → new_state
        inputs: tuple of tensors, each shape (T, batch, ...)
        start:  initial state (dict or tuple of tensors)

    Returns:
        List containing the stacked output states over time.
    """
    last = start
    indices = range(inputs[0].shape[0])
    flag = True
    for index in indices:
        last = fn(last, *[inp[index] for inp in inputs])
        if flag:
            if isinstance(last, dict):
                outputs = {k: v.clone().unsqueeze(0) for k, v in last.items()}
            else:
                outputs = []
                for _last in last:
                    if isinstance(_last, dict):
                        outputs.append({k: v.clone().unsqueeze(0) for k, v in _last.items()})
                    else:
                        outputs.append(_last.clone().unsqueeze(0))
            flag = False
        else:
            if isinstance(last, dict):
                for k in last:
                    outputs[k] = torch.cat([outputs[k], last[k].unsqueeze(0)], dim=0)
            else:
                for j in range(len(outputs)):
                    if isinstance(last[j], dict):
                        for k in last[j]:
                            outputs[j][k] = torch.cat(
                                [outputs[j][k], last[j][k].unsqueeze(0)], dim=0
                            )
                    else:
                        outputs[j] = torch.cat(
                            [outputs[j], last[j].unsqueeze(0)], dim=0
                        )
    if isinstance(last, dict):
        outputs = [outputs]
    return outputs


# ── Distribution wrappers ──────────────────────────────────────────────────

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class ContDist:
    """Wraps a continuous torch distribution; adds mode() with optional absmax clipping."""
    def __init__(self, dist=None, absmax=None):
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out = out * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out = out * (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """OneHotCategorical with optional uniform mixing and straight-through gradients."""
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(torch.argmax(super().logits, dim=-1), super().logits.shape[-1])
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        sample = super().sample(sample_shape).detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample = sample + probs - probs.detach()
        return sample


class MSEDist:
    """MSE reconstruction distribution (negative MSE as log_prob)."""
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        dims = list(range(len(distance.shape)))[2:]
        if self._agg == "mean":
            return -distance.mean(dims)
        elif self._agg == "sum":
            return -distance.sum(dims)
        raise NotImplementedError(self._agg)


class SymlogDist:
    """Symlog-space MSE distribution."""
    def __init__(self, mode, agg="sum", tol=1e-8):
        self._mode = mode
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        distance = (self._mode - symlog(value)) ** 2.0
        distance = torch.where(distance < self._tol, torch.zeros_like(distance), distance)
        dims = list(range(len(distance.shape)))[2:]
        if self._agg == "mean":
            return -distance.mean(dims)
        elif self._agg == "sum":
            return -distance.sum(dims)
        raise NotImplementedError(self._agg)


class SafeTruncatedNormal(torchd.normal.Normal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale)
        self._low = low
        self._high = high
        self._clip = clip
        self._mult = mult

    def sample(self, sample_shape):
        event = super().sample(sample_shape)
        if self._clip:
            clipped = torch.clip(event, self._low + self._clip, self._high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event = event * self._mult
        return event


# ── Building block layers ──────────────────────────────────────────────────

class GRUCell(nn.Module):
    """Normed GRU cell identical to dreamerv3-torch networks.GRUCell."""
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super().__init__()
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential()
        self.layers.add_module("GRU_linear", nn.Linear(inp_size + size, 3 * size, bias=False))
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-3))

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]
        parts = self.layers(torch.cat([inputs, state], -1))
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(nn.Conv2d):
    """Conv2d with same-padding (no size reduction unless stride > 1)."""
    def _calc_same_pad(self, i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ImgChLayerNorm(nn.Module):
    """Channel-wise LayerNorm for NCHW feature maps."""
    def __init__(self, ch, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


# ── Optimizer wrapper ──────────────────────────────────────────────────────

class Optimizer:
    """
    Thin wrapper around torch.optim with grad clipping, weight decay, and AMP.
    Adapted from dreamerv3-torch tools.Optimizer.
    """
    def __init__(self, name, parameters, lr, eps=1e-4, clip=None, wd=0.0, opt="adam", use_amp=False):
        assert 0 <= wd < 1
        self._name = name
        self._clip = clip
        self._wd = wd
        self._opt = {
            "adam": lambda: torch.optim.Adam(parameters, lr=lr, eps=eps),
            "adamax": lambda: torch.optim.Adamax(parameters, lr=lr, eps=eps),
            "sgd": lambda: torch.optim.SGD(parameters, lr=lr),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def __call__(self, loss, params, retain_graph=True):
        assert loss.ndim == 0, loss.shape
        metrics = {f"{self._name}_loss": loss.detach().cpu().item()}
        self._opt.zero_grad()
        self._scaler.scale(loss).backward(retain_graph=retain_graph)
        self._scaler.unscale_(self._opt)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip or 1e9)
        if self._wd:
            for var in params:
                var.data = (1 - self._wd) * var.data
        self._scaler.step(self._opt)
        self._scaler.update()
        self._opt.zero_grad()
        metrics[f"{self._name}_grad_norm"] = norm.item()
        return metrics


# ── Misc ───────────────────────────────────────────────────────────────────

class RequiresGrad:
    def __init__(self, model):
        self._model = model

    def __enter__(self):
        self._model.requires_grad_(True)

    def __exit__(self, *args):
        self._model.requires_grad_(False)


if __name__ == "__main__":
    # Smoke: weight_init on a tiny linear
    lin = nn.Linear(16, 16)
    lin.apply(weight_init)

    # Smoke: static_scan over T=5 steps
    T, B, D = 5, 2, 8
    state = {"h": torch.zeros(B, D)}

    def step(prev, inp):
        return {"h": prev["h"] + inp}

    inputs = (torch.ones(T, B, D),)
    out = static_scan(step, inputs, state)
    assert out[0]["h"].shape == (T, B, D), out[0]["h"].shape

    # Smoke: GRUCell
    cell = GRUCell(inp_size=4, size=8)
    x = torch.randn(B, 4)
    s = [torch.zeros(B, 8)]
    y, s2 = cell(x, s)
    assert y.shape == (B, 8)

    # Smoke: OneHotDist
    logits = torch.randn(B, 32, 32)
    dist = OneHotDist(logits=logits, unimix_ratio=0.01)
    sample = dist.sample()
    assert sample.shape == (B, 32, 32)

    print("dreamer_utils: all smoke checks passed.")
