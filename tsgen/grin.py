"""GRIN (Cini, Marisca & Alippi, ICLR 2022): re-implementation from the paper.

The official code has no licence, so this module is written from the paper's equations,
with dense adjacency (N ≤ 36 here). Per direction, a Graph Recurrent Imputation Layer:

    Ŷ¹_t = H_{t-1} V₁                          first-stage imputation
    X̂¹_t = M_t ⊙ X_t + (1 − M_t) ⊙ Ŷ¹_t
    S_t   = MPNN_{no self}([X̂¹_t ‖ M_t ‖ H_{t-1}])  spatial decoder (neighbours only)
    Ŷ²_t = [S_t ‖ H_{t-1}] V₂                  second-stage imputation
    X̂²_t = M_t ⊙ X_t + (1 − M_t) ⊙ Ŷ²_t
    H_t   = MPGRU([X̂²_t ‖ M_t ‖ U_t], H_{t-1})  GRU whose gates are order-K diffusion convolutions

Forward and backward layers are merged by an MLP on [S ‖ H] of both directions. The loss
is the MAE of the final and of the four intermediate imputations on the observed entries;
during training a further fraction of observed inputs is hidden at random. GRIN reads the
whole window, future included: it is a non-causal imputer.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import Series, gather_windows

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrinConfig:
    window: int = 24
    hidden: int = 64
    ff: int = 64
    order: int = 2
    epochs: int = 300
    patience: int = 40
    batch_size: int = 32
    lr: float = 1e-3
    whiten: float = 0.05
    seed: int = 0
    device: str = "cpu"


def supports(adj: torch.Tensor, self_loops: bool) -> list[torch.Tensor]:
    """Row-normalised forward and backward transition matrices."""
    a = adj.clone()
    if not self_loops:
        a.fill_diagonal_(0)
    out = []
    for m in (a, a.t()):
        out.append(m / m.sum(-1, keepdim=True).clamp_min(1e-8))
    return out


class DiffusionConv(nn.Module):
    """Σ_k Σ_s P_s^k X Θ_{s,k} (+ X Θ₀ when include_self) on dense supports."""

    def __init__(self, fin: int, fout: int, order: int, n_supports: int = 2, include_self: bool = True):
        super().__init__()
        self.order, self.include_self = order, include_self
        n_terms = n_supports * order + int(include_self)
        self.lin = nn.Linear(fin * n_terms, fout)

    def forward(self, x: torch.Tensor, sup: list[torch.Tensor]) -> torch.Tensor:
        """x: (B, N, F)."""
        terms = [x] if self.include_self else []
        for p in sup:
            z = x
            for _ in range(self.order):
                z = torch.einsum("ij,bjf->bif", p, z)
                terms.append(z)
        return self.lin(torch.cat(terms, -1))


class MPGRUCell(nn.Module):
    def __init__(self, fin: int, hidden: int, order: int):
        super().__init__()
        self.gates = DiffusionConv(fin + hidden, 2 * hidden, order)
        self.cand = DiffusionConv(fin + hidden, hidden, order)

    def forward(self, x, h, sup):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), sup)).chunk(2, -1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), sup))
        return u * h + (1 - u) * c


class GRIL(nn.Module):
    def __init__(self, n_exo: int, hidden: int, order: int):
        super().__init__()
        self.hidden = hidden
        self.first = nn.Linear(hidden, 1)
        self.decoder = DiffusionConv(2 + hidden, hidden, 1, include_self=False)
        self.second = nn.Linear(2 * hidden, 1)
        self.cell = MPGRUCell(2 + n_exo, hidden, order)
        self.h0 = nn.Parameter(torch.zeros(hidden))

    def forward(self, x, m, u, sup, sup_noself):
        """x, m: (B, T, N); u: (B, T, E). Returns per-step (ŷ¹, ŷ², S, H_{t-1}) stacked over T."""
        b, t_len, n = x.shape
        h = self.h0.expand(b, n, self.hidden)
        outs = {k: [] for k in ("y1", "y2", "s", "h")}
        for t in range(t_len):
            xt, mt = x[:, t, :, None], m[:, t, :, None]
            y1 = self.first(h)
            x1 = mt * xt + (1 - mt) * y1
            s = torch.relu(self.decoder(torch.cat([x1, mt, h], -1), sup_noself))
            y2 = self.second(torch.cat([s, h], -1))
            x2 = mt * xt + (1 - mt) * y2
            ut = u[:, t, None, :].expand(b, n, u.shape[-1])
            for k, v in (("y1", y1), ("y2", y2), ("s", s), ("h", h)):
                outs[k].append(v)
            h = self.cell(torch.cat([x2, mt, ut], -1), h, sup)
        return {k: torch.stack(v, 1) for k, v in outs.items()}  # (B, T, N, ·)


class GRIN(nn.Module):
    def __init__(self, adj: torch.Tensor, n_exo: int, hidden: int = 64, ff: int = 64, order: int = 2):
        super().__init__()
        self.register_buffer("adj", adj.float())
        self.fwd, self.bwd = GRIL(n_exo, hidden, order), GRIL(n_exo, hidden, order)
        self.merge = nn.Sequential(nn.Linear(4 * hidden, ff), nn.ReLU(), nn.Linear(ff, 1))

    def forward(self, x, m, u):
        m = m.to(x.dtype)
        sup, sup_ns = supports(self.adj, True), supports(self.adj, False)
        f = self.fwd(x * m, m, u, sup, sup_ns)
        r = self.bwd(*(v.flip(1) for v in (x * m, m, u)), sup, sup_ns)
        r = {k: v.flip(1) for k, v in r.items()}
        y = self.merge(torch.cat([f["s"], f["h"], r["s"], r["h"]], -1))[..., 0]
        return y, [f["y1"][..., 0], f["y2"][..., 0], r["y1"][..., 0], r["y2"][..., 0]]


def _masked_mae(pred, target, mask):
    w = mask.to(pred.dtype)
    return ((pred - target).abs() * w).sum() / w.sum().clamp_min(1.0)


def _windows(series: Series, lo: int, hi: int, w: int) -> np.ndarray:
    return np.arange(max(lo, 0), hi - w + 1)


def _tensors(series: Series, starts: np.ndarray, w: int):
    x = torch.from_numpy(gather_windows(series.values, starts, w))
    m = torch.from_numpy(gather_windows(series.mask, starts, w))
    u = torch.from_numpy(gather_windows(series.exo, starts, w))
    return x, m, u


def _val_score(model, series: Series, cfg: GrinConfig, device) -> float:
    """MAE on the held-out entries (eval_mask) of the validation range; input hides them."""
    lo, hi = series.split_range("val")
    filled = impute_range(model, series, cfg, lo, hi)
    hidden = np.zeros_like(series.eval_mask)
    hidden[lo:hi] = series.eval_mask[lo:hi]
    return float(np.abs(filled - np.nan_to_num(series.truth))[hidden].mean())


def fit(series: Series, adj: np.ndarray, cfg: GrinConfig) -> GRIN:
    """Train on training windows; early stopping on validation held-out MAE (best weights restored)."""
    if series.eval_mask is None:
        raise ValueError("GRIN needs held-out entries (eval_mask) for validation and scoring")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = GRIN(torch.from_numpy(adj), series.n_exo, cfg.hidden, cfg.ff, cfg.order).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x_all, m_all, u_all = _tensors(series, _windows(series, 0, series.split_range("train")[1], cfg.window), cfg.window)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_all), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x, m, u = x_all[idx].to(device), m_all[idx].to(device), u_all[idx].to(device)
            keep = (torch.rand(m.shape, generator=gen) >= cfg.whiten).to(device)
            y, parts = model(x, m & keep, u)
            loss = _masked_mae(y, x, m) + sum(_masked_mae(p, x, m) for p in parts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        score = _val_score(model, series, cfg, device)
        log.info("grin epoch %d val %.5f", epoch, score)
        if score < best - 1e-6:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.best_val, model.epochs_run = best, epoch + 1
    return model


@torch.no_grad()
def impute_range(model: GRIN, series: Series, cfg: GrinConfig, lo: int, hi: int, batch: int = 256) -> np.ndarray:
    """Fill hidden entries in [lo, hi): stride-1 windows covering the range, predictions averaged."""
    model.eval()
    device = next(model.parameters()).device
    w = cfg.window
    starts = np.arange(max(lo - w + 1, 0), min(hi, len(series.values) - w + 1))
    acc = np.zeros(series.values.shape, np.float64)
    cnt = np.zeros(series.values.shape, np.float64)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        x, m, u = _tensors(series, s, w)
        y, _ = model(x.to(device), m.to(device), u.to(device))
        idx = s[:, None] + np.arange(w)[None, :]
        np.add.at(acc, idx, y.cpu().numpy())
        np.add.at(cnt, idx, 1.0)
    filled = series.values.copy()
    hidden = ~series.mask
    rows = np.zeros(len(filled), bool)
    rows[lo:hi] = True
    sel = hidden & rows[:, None] & (cnt > 0)
    filled[sel] = (acc[sel] / cnt[sel]).astype(np.float32)
    return filled
