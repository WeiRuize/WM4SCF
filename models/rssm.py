"""
rssm.py
=======
Recurrent State-Space Model adapted from:
    dreamerv3-torch-main/networks.py  class RSSM

Changes vs. upstream:
  - Removed dependency on dreamer `tools` module; uses local dreamer_utils.
  - Added `encode_sequence()` convenience wrapper.
  - Added `rollout_future()` for Safety Guardian use.
  - Docstrings added.

Usage in SafeVLA:
  - Trained jointly with RobotObsEncoder and RobotObsDecoder (train_rssm.py).
  - Loaded frozen (requires_grad=False) by SafetyGuardian at inference time.
  - Do NOT call img_step() with imagined states to update the history buffer
    in Guardian; always feed real posterior states back.
"""

import torch
from torch import nn

from models.dreamer_utils import (
    GRUCell,
    OneHotDist,
    ContDist,
    SafeTruncatedNormal,
    static_scan,
    weight_init,
    uniform_weight_init,
)
from torch import distributions as torchd


class RSSM(nn.Module):
    """
    Recurrent State-Space Model with optional discrete stochastic states.

    State dict keys:
        Discrete mode  → {logit, stoch, deter}
        Continuous mode → {mean, std, stoch, deter}

    Feature vector: feat = cat(stoch.reshape(-1), deter)
    """

    def __init__(
        self,
        stoch: int = 32,
        deter: int = 512,
        hidden: int = 512,
        rec_depth: int = 1,
        discrete: int = 32,     # 0 for continuous
        act: str = "SiLU",
        norm: bool = True,
        mean_act: str = "none",
        std_act: str = "sigmoid2",
        min_std: float = 0.1,
        unimix_ratio: float = 0.01,
        initial: str = "learned",
        num_actions: int = 7,
        embed: int = None,      # encoder output dim; required
        device: str = "cuda:0",
    ):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device

        act_cls = getattr(nn, act)

        # img_in: [stoch; action] → hidden  (prior / imagination path)
        inp_dim = (stoch * discrete if discrete else stoch) + num_actions
        img_in = []
        img_in.append(nn.Linear(inp_dim, hidden, bias=False))
        if norm:
            img_in.append(nn.LayerNorm(hidden, eps=1e-3))
        img_in.append(act_cls())
        self._img_in_layers = nn.Sequential(*img_in)
        self._img_in_layers.apply(weight_init)

        self._cell = GRUCell(hidden, deter, norm=norm)
        self._cell.apply(weight_init)

        # img_out: deter → hidden  (prior statistics)
        img_out = []
        img_out.append(nn.Linear(deter, hidden, bias=False))
        if norm:
            img_out.append(nn.LayerNorm(hidden, eps=1e-3))
        img_out.append(act_cls())
        self._img_out_layers = nn.Sequential(*img_out)
        self._img_out_layers.apply(weight_init)

        # obs_out: [deter; embed] → hidden  (posterior statistics)
        obs_out = []
        obs_out.append(nn.Linear(deter + embed, hidden, bias=False))
        if norm:
            obs_out.append(nn.LayerNorm(hidden, eps=1e-3))
        obs_out.append(act_cls())
        self._obs_out_layers = nn.Sequential(*obs_out)
        self._obs_out_layers.apply(weight_init)

        # Stochastic stat layers
        if discrete:
            self._imgs_stat_layer = nn.Linear(hidden, stoch * discrete)
            self._obs_stat_layer = nn.Linear(hidden, stoch * discrete)
        else:
            self._imgs_stat_layer = nn.Linear(hidden, 2 * stoch)
            self._obs_stat_layer = nn.Linear(hidden, 2 * stoch)
        self._imgs_stat_layer.apply(uniform_weight_init(1.0))
        self._obs_stat_layer.apply(uniform_weight_init(1.0))

        if initial == "learned":
            self.W = nn.Parameter(
                torch.zeros((1, deter), device=device), requires_grad=True
            )

    # ── Initial state ──────────────────────────────────────────────────────

    def initial(self, batch_size: int) -> dict:
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = dict(
                logit=torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
                stoch=torch.zeros(batch_size, self._stoch, self._discrete, device=self._device),
                deter=deter,
            )
        else:
            state = dict(
                mean=torch.zeros(batch_size, self._stoch, device=self._device),
                std=torch.zeros(batch_size, self._stoch, device=self._device),
                stoch=torch.zeros(batch_size, self._stoch, device=self._device),
                deter=deter,
            )
        if self._initial == "zeros":
            return state
        elif self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        raise NotImplementedError(self._initial)

    # ── Forward passes ─────────────────────────────────────────────────────

    def observe(self, embed, action, is_first, state=None):
        """
        Run posterior update over a full (batch, time) sequence.

        Args:
            embed:    (B, T, embed_dim)
            action:   (B, T, action_dim)
            is_first: (B, T)
            state:    initial RSSM state dict; None → use self.initial()

        Returns:
            post: dict of (B, T, ...) posterior states
            prior: dict of (B, T, ...) prior states
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        if state is None:
            state = self.initial(embed.shape[1])

        post, prior = static_scan(
            lambda prev_state, prev_act, emb, is_f: self.obs_step(prev_state[0], prev_act, emb, is_f),
            (action, embed, is_first),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        """
        Unroll prior (imagination) steps given a sequence of actions.

        Args:
            action: (B, T, action_dim)
            state:  starting RSSM state dict (B, ...)

        Returns:
            prior: dict of (B, T, ...) imagined states
        """
        swap = lambda x: x.permute([1, 0] + list(range(2, x.ndim)))
        action = swap(action)
        prior = static_scan(self.img_step, [action], state)
        prior = prior[0]
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        """Single-step posterior update."""
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros(len(is_first), self._num_actions, device=self._device)
        elif torch.sum(is_first) > 0:
            is_first_col = is_first[:, None]
            prev_action = prev_action * (1.0 - is_first_col)
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = is_first_col.reshape(
                    is_first_col.shape + (1,) * (val.ndim - is_first_col.ndim)
                )
                prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer("obs", x)
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        """Single-step prior update (imagination)."""
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = prev_stoch.reshape(list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete])
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in_layers(x)
        deter = prev_state["deter"]
        for _ in range(self._rec_depth):
            x, deter = self._cell(x, [deter])
            deter = deter[0]
        x = self._img_out_layers(x)
        stats = self._suff_stats_layer("ims", x)
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        return {"stoch": stoch, "deter": deter, **stats}

    # ── Feature / distribution helpers ────────────────────────────────────

    def get_feat(self, state: dict) -> torch.Tensor:
        """Concatenate stoch and deter into a flat feature vector."""
        stoch = state["stoch"]
        if self._discrete:
            stoch = stoch.reshape(list(stoch.shape[:-2]) + [self._stoch * self._discrete])
        return torch.cat([stoch, state["deter"]], -1)

    def get_dist(self, state: dict):
        if self._discrete:
            return OneHotDist(logits=state["logit"], unimix_ratio=self._unimix_ratio)
        mean, std = state["mean"], state["std"]
        return ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))

    def get_stoch(self, deter: torch.Tensor) -> torch.Tensor:
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer("ims", x)
        return self.get_dist(stats).mode()

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        kld = torchd.kl.kl_divergence
        dist = self.get_dist
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        rep_loss_raw = kld(
            dist(post) if self._discrete else dist(post)._dist,
            dist(sg(prior)) if self._discrete else dist(sg(prior))._dist,
        )
        dyn_loss_raw = kld(
            dist(sg(post)) if self._discrete else dist(sg(post))._dist,
            dist(prior) if self._discrete else dist(prior)._dist,
        )
        kl_value = rep_loss_raw                               # raw KL for logging
        rep_loss = torch.clip(rep_loss_raw, min=free)
        dyn_loss = torch.clip(dyn_loss_raw, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        # Returns: (total_loss, kl_value_for_logging, dyn_loss, rep_loss)
        return loss, kl_value, dyn_loss, rep_loss

    # ── Convenience wrappers for SafeVLA ──────────────────────────────────

    def encode_sequence(self, embeds, actions, is_first, state=None):
        """Alias for observe(); returns only the posterior states."""
        post, _ = self.observe(embeds, actions, is_first, state)
        return post

    def rollout_future(self, last_state: dict, horizon: int, action_policy=None) -> dict:
        """
        Unroll imagination from last_state for `horizon` steps.
        If action_policy is None, uses zero actions (conservative).

        Args:
            last_state:    RSSM state dict, values of shape (B, ...)
            horizon:       number of future steps to predict
            action_policy: optional callable(feat) → action tensor, or None

        Returns:
            imagined: dict of (B, horizon, ...) imagined states
        """
        B = last_state["deter"].shape[0]
        device = last_state["deter"].device
        state = last_state
        states = []
        for _ in range(horizon):
            feat = self.get_feat(state)
            if action_policy is not None:
                action = action_policy(feat)
            else:
                action = torch.zeros(B, self._num_actions, device=device)
            state = self.img_step(state, action, sample=False)
            states.append(state)

        # Stack over time dimension
        imagined = {
            k: torch.stack([s[k] for s in states], dim=1)
            for k in states[0]
        }
        return imagined

    # ── Internal ──────────────────────────────────────────────────────────

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            layer = self._imgs_stat_layer if name == "ims" else self._obs_stat_layer
            x = layer(x)
            logit = x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])
            return {"logit": logit}
        else:
            layer = self._imgs_stat_layer if name == "ims" else self._obs_stat_layer
            x = layer(x)
            mean, std = torch.split(x, [self._stoch] * 2, -1)
            mean = {"none": mean, "tanh5": 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]
            std_map = {
                "softplus": lambda: torch.nn.functional.softplus(std),
                "abs": lambda: torch.abs(std + 1),
                "sigmoid": lambda: torch.sigmoid(std),
                "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
            }
            std = std_map[self._std_act]() + self._min_std
            return {"mean": mean, "std": std}


if __name__ == "__main__":
    device = "cpu"
    B, T, A, E = 2, 4, 7, 256
    rssm = RSSM(stoch=4, deter=16, hidden=16, discrete=4, num_actions=A, embed=E, device=device)
    rssm.to(device)

    embed = torch.randn(B, T, E)
    action = torch.randn(B, T, A)
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    post, prior = rssm.observe(embed, action, is_first)
    feat_size = 4 * 4 + 16  # stoch*discrete + deter = 32
    assert rssm.get_feat(post).shape == (B, T, feat_size), rssm.get_feat(post).shape

    # rollout future
    last_state = {k: v[:, -1] for k, v in post.items()}
    imagined = rssm.rollout_future(last_state, horizon=3)
    assert imagined["deter"].shape == (B, 3, 16), imagined["deter"].shape

    print("rssm: all smoke checks passed.")
