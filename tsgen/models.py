"""Graph mixture-density generators (MDN, MR, SMR, ASMR) and point baselines.

MR = Multivariate Regressor (temporal visibility-graph block), SMR = Spatial MR (+ static
spatial graph), ASMR = Adaptive Spatial MR (+ learned spatial graph); formerly GTM, SGTM, ASGTM.

One conditional model p(x_{t+1} | x_{≤t}, y_{≤t}) with switchable blocks:

    F_t   = [x_t − x_s ‖ y_t ‖ m_t]                    inputs relative to the window's first step
                                                       (+ observation mask)
    T_t   = ReLU(Σ_k (P^k F)_t Θ_k)                    causal temporal diffusion on the window graph P
    S_t   = ReLU(Σ_k Ã^k X_t Θ^f_k + (Ãᵀ)^k X_t Θ^b_k)  spatial diffusion per time step, sensors only
    h_t   = LSTM([F_t ‖ T_t ‖ vec S_t])                unidirectional
    p(x_{t+1}) = Σ_m π_m(h_t) Π_i N(x^i | x^i_t + Δμ^i_m(h_t), σ^i_m(h_t)²)

Relative inputs and last-value-anchored means (`relative=True`) make the model
invariant to the level of the series: needed for non-stationary data such as
exchange rates, whose validation/test range lies outside the training range.
With Δμ = 0 the mixture mean is the persistence forecast. Relative inputs alone hide
the absolute level, so a stationary series cannot mean-revert in free-running
generation; `level_input=True` also feeds x_t itself. Every block only reads steps ≤ t, so teacher-forced training has no future leakage
and matches free-running generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = {
    #          temporal  spatial
    "mdn":   (False, None),
    "mr":    (True, None),
    "smr":   (True, "static"),
    "asmr":  (True, "adaptive"),
}
ALIASES = {"gtm": "mr", "sgtm": "smr", "asgtm": "asmr"}  # names used before 2026-10-01


def canonical(name: str) -> str:
    return ALIASES.get(name, name)
LOG_2PI = math.log(2 * math.pi)
MIN_SIGMA = 1e-3


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "asmr"
    hidden: int = 64
    mixtures: int = 8
    temporal_hidden: int = 16
    spatial_hidden: int = 4
    hops: int = 2
    embedding: int = 8
    topk: int = 5
    use_mask: bool = False
    relative: bool = True
    level_input: bool = False  # also feed absolute levels, so the model can mean-revert


class MixtureHead(nn.Module):
    """Diagonal Gaussian mixture over N variables: π ∈ Δ^M, μ, σ ∈ R^{M×N}."""

    def __init__(self, hidden: int, n: int, m: int):
        super().__init__()
        self.n, self.m = n, m
        self.proj = nn.Linear(hidden, m + 2 * m * n)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(h)
        logits, mu, s = out.split([self.m, self.m * self.n, self.m * self.n], dim=-1)
        shape = (*h.shape[:-1], self.m, self.n)
        sigma = F.softplus(s.reshape(shape)) + MIN_SIGMA
        return F.log_softmax(logits, -1), mu.reshape(shape), sigma

    @staticmethod
    def component_loglik(params, y: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """log π_m + Σ_i log N(y_i | μ_mi, σ_mi²) over observed i → (..., M)."""
        log_pi, mu, sigma = params
        z = (y.unsqueeze(-2) - mu) / sigma
        ll = -0.5 * (LOG_2PI + z * z) - sigma.log()
        if obs is not None:
            ll = ll * obs.unsqueeze(-2).to(ll.dtype)  # missing dims are marginalised out exactly
        return log_pi + ll.sum(-1)

    def nll(self, params, y: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """Mean of −log p(y) over positions with at least one observed target."""
        lp = torch.logsumexp(self.component_loglik(params, y, obs), dim=-1)
        if obs is None:
            return -lp.mean()
        keep = obs.any(-1).to(lp.dtype)
        return -(lp * keep).sum() / keep.sum().clamp_min(1.0)

    @staticmethod
    def sample(params, generator: torch.Generator | None = None) -> torch.Tensor:
        """k ~ Cat(π), then x ~ N(μ_k, diag σ_k²)."""
        log_pi, mu, sigma = params
        k = torch.multinomial(log_pi.exp().reshape(-1, log_pi.shape[-1]), 1, generator=generator)
        k = k.reshape(*log_pi.shape[:-1], 1, 1).expand(*mu.shape[:-2], 1, mu.shape[-1])
        mu_k, sigma_k = mu.gather(-2, k).squeeze(-2), sigma.gather(-2, k).squeeze(-2)
        eps = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
        return mu_k + sigma_k * eps

    @staticmethod
    def mean(params) -> torch.Tensor:
        log_pi, mu, _ = params
        return (log_pi.exp().unsqueeze(-1) * mu).sum(-2)

    def conditional_mean(self, params, y: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """E[x_missing | x_observed] at the same step: posterior component weights × μ."""
        _, mu, _ = params
        post = torch.softmax(self.component_loglik(params, y, obs), dim=-1)
        return (post.unsqueeze(-1) * mu).sum(-2)


class TemporalDiffusion(nn.Module):
    """Z = ReLU(Σ_{k=0..K} P^k F Θ_k), P causal (B, w, w). Forward-in-time only."""

    def __init__(self, fin: int, fout: int, hops: int):
        super().__init__()
        self.lins = nn.ModuleList(nn.Linear(fin, fout, bias=k == 0) for k in range(hops + 1))

    def forward(self, f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        out, z = self.lins[0](f), f
        for lin in self.lins[1:]:
            z = torch.bmm(p, z)
            out = out + lin(z)
        return F.relu(out)


class SpatialDiffusion(nn.Module):
    """Per-step bidirectional diffusion over sensors (DCRNN), node features = [x_ti, m_ti]."""

    def __init__(self, fin: int, fout: int, hops: int):
        super().__init__()
        self.root = nn.Linear(fin, fout)
        self.fwd = nn.ModuleList(nn.Linear(fin, fout, bias=False) for _ in range(hops))
        self.bwd = nn.ModuleList(nn.Linear(fin, fout, bias=False) for _ in range(hops))

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """x: (B, w, N, fin), a: (N, N) row-normalised → (B, w, N, fout)."""
        a_b = a.t() / a.t().sum(-1, keepdim=True).clamp_min(1e-8)
        out, zf, zb = self.root(x), x, x
        for lf, lb in zip(self.fwd, self.bwd):
            zf, zb = torch.einsum("ij,bwjf->bwif", a, zf), torch.einsum("ij,bwjf->bwif", a_b, zb)
            out = out + lf(zf) + lb(zb)
        return F.relu(out)


class AdaptiveAdjacency(nn.Module):
    """Ã = softmax over the top-k entries of ReLU(E₁E₂ᵀ) per row (others exactly 0)."""

    def __init__(self, n: int, dim: int, topk: int):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n, dim) / math.sqrt(dim))
        self.e2 = nn.Parameter(torch.randn(n, dim) / math.sqrt(dim))
        self.topk = min(topk, n - 1)

    def forward(self) -> torch.Tensor:
        s = F.relu(self.e1 @ self.e2.t())
        s = s.masked_fill(torch.eye(len(s), dtype=torch.bool, device=s.device), float("-inf"))
        idx = s.topk(self.topk, dim=-1).indices
        keep = torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True)
        return torch.softmax(s.masked_fill(~keep, float("-inf")), dim=-1)


class GraphMixtureGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig, n: int, e: int, static_adj: torch.Tensor | None = None):
        super().__init__()
        cfg = ModelConfig(**{**cfg.__dict__, "variant": canonical(cfg.variant)})
        if cfg.variant not in VARIANTS:
            raise ValueError(f"unknown variant '{cfg.variant}' {tuple(VARIANTS)}")
        temporal, spatial = VARIANTS[cfg.variant]
        if spatial == "static" and static_adj is None:
            raise ValueError("smr needs a static adjacency")
        self.cfg, self.n, self.e = cfg, n, e
        self.needs_temporal_graph = temporal
        fin = n * (2 if cfg.level_input else 1) + e + (n if cfg.use_mask else 0)
        feats = fin
        self.temporal = TemporalDiffusion(fin, cfg.temporal_hidden, cfg.hops) if temporal else None
        feats += cfg.temporal_hidden if temporal else 0
        self.spatial, self.adaptive = None, None
        if spatial:
            node_in = 1 + int(cfg.use_mask) + int(cfg.level_input)
            self.spatial = SpatialDiffusion(node_in, cfg.spatial_hidden, cfg.hops)
            feats += n * cfg.spatial_hidden
            if spatial == "adaptive":
                self.adaptive = AdaptiveAdjacency(n, cfg.embedding, cfg.topk)
        if static_adj is not None:
            static = static_adj / static_adj.sum(-1, keepdim=True).clamp_min(1e-8)
            self.register_buffer("static_adj", static.float())
        else:
            self.static_adj = None
        self.lstm = nn.LSTM(feats, cfg.hidden, batch_first=True)
        self.head = MixtureHead(cfg.hidden, n, cfg.mixtures)

    def adjacency(self) -> torch.Tensor | None:
        if self.adaptive is not None:
            return self.adaptive()
        return self.static_adj

    def encode(self, x, exo, obs=None, p_time=None) -> torch.Tensor:
        """x (B, w, N), exo (B, w, E), obs (B, w, N) bool, p_time (B, w, w) → h (B, w, H)."""
        xr = x - x[:, :1] if self.cfg.relative else x
        parts = [xr, x, exo] if self.cfg.level_input else [xr, exo]
        if self.cfg.use_mask:
            parts.append(obs.to(x.dtype))
        f = torch.cat(parts, -1)
        feats = [f]
        if self.temporal is not None:
            if p_time is None:
                raise ValueError(f"variant '{self.cfg.variant}' needs the temporal graph p_time")
            feats.append(self.temporal(f, p_time))
        if self.spatial is not None:
            node = xr.unsqueeze(-1)
            if self.cfg.level_input:
                node = torch.cat([node, x.unsqueeze(-1)], -1)
            if self.cfg.use_mask:
                node = torch.cat([node, obs.to(x.dtype).unsqueeze(-1)], -1)
            s = self.spatial(node, self.adjacency())
            feats.append(s.flatten(-2))
        h, _ = self.lstm(torch.cat(feats, -1))
        return h

    def forward(self, x, exo, obs=None, p_time=None):
        log_pi, mu, sigma = self.head(self.encode(x, exo, obs, p_time))
        if self.cfg.relative:
            mu = mu + x.unsqueeze(-2)
        return log_pi, mu, sigma

    def loss(self, batch) -> torch.Tensor:
        return self.head.nll(self(batch.x, batch.exo, batch.obs, batch.p_time), batch.y, batch.y_obs)


class PointForecaster(nn.Module):
    """LSTM / RNN one-step point forecaster (MSE), same windows and inputs as the generators."""

    def __init__(self, cell: str, n: int, e: int, hidden: int = 64, use_mask: bool = False,
                 relative: bool = True):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "rnn": nn.RNN}
        if cell not in rnn:
            raise ValueError(f"unknown cell '{cell}' (lstm, rnn)")
        self.use_mask, self.relative = use_mask, relative
        self.needs_temporal_graph = False
        self.rnn = rnn[cell](n + e + (n if use_mask else 0), hidden, batch_first=True)
        self.out = nn.Linear(hidden, n)

    def forward(self, x, exo, obs=None, p_time=None) -> torch.Tensor:
        xr = x - x[:, :1] if self.relative else x
        parts = [xr, exo] + ([obs.to(x.dtype)] if self.use_mask else [])
        h, _ = self.rnn(torch.cat(parts, -1))
        return self.out(h) + (x if self.relative else 0)

    def loss(self, batch) -> torch.Tensor:
        err = (self(batch.x, batch.exo, batch.obs) - batch.y) ** 2
        w = batch.y_obs.to(err.dtype) if batch.y_obs is not None else torch.ones_like(err)
        return (err * w).sum() / w.sum().clamp_min(1.0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(cfg: ModelConfig, n: int, e: int, static_adj=None, param_budget: int | None = None) -> nn.Module:
    """Build a generator; with `param_budget`, pick the LSTM width whose total size is closest."""
    if param_budget is None:
        return GraphMixtureGenerator(cfg, n, e, static_adj)
    best = None
    for hidden in range(8, 513, 4):
        model = GraphMixtureGenerator(ModelConfig(**{**cfg.__dict__, "hidden": hidden}), n, e, static_adj)
        gap = abs(count_params(model) - param_budget)
        if best is None or gap < best[0]:
            best = (gap, model)
    return best[1]
