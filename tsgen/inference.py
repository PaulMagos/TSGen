"""Prediction, causal imputation and autoregressive generation with one model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import Series, gather_windows, window_starts
from .graphs import temporal_adjacency
from .models import MixtureHead
from .train import TrainConfig, make_batch


@dataclass(frozen=True)
class Forecast:
    targets: np.ndarray        # (T_split,) absolute time indices
    y_true: np.ndarray         # (T_split, N) scaled truth, NaN where never observed
    y_pred: np.ndarray         # (T_split, N) point forecast (mixture mean)
    samples: np.ndarray | None  # (S, T_split, N) predictive samples, for CRPS


def _is_mixture(model) -> bool:
    return hasattr(model, "head")


def _params_last(model, x, exo, obs, cfg: TrainConfig, device):
    """Distribution parameters (or point output) for the step after the last input."""
    p = None
    if getattr(model, "needs_temporal_graph", False):
        p = torch.from_numpy(temporal_adjacency(x.numpy(), cfg.temporal_graph, cfg.edge_weight)).to(device)
    out = model(x.to(device), exo.to(device), obs.to(device), p)
    if isinstance(out, tuple):
        return tuple(o[:, -1] for o in out)
    return out[:, -1]


@torch.no_grad()
def predict(model, series: Series, cfg: TrainConfig, split: str = "test", n_samples: int = 100,
            batch_size: int = 512, seed: int = 0) -> Forecast:
    """One-step-ahead forecasts x̂_t from the true window x[t-w:t], for every t in the split."""
    model.eval()
    device = torch.device(cfg.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    starts = window_starts(series, split, cfg.window)
    preds, draws = [], []
    for i in range(0, len(starts), batch_size):
        b = make_batch(series, starts[i:i + batch_size], cfg.window, None)
        out = _params_last(model, b.x, b.exo, b.obs, cfg, device)
        if _is_mixture(model):
            cpu = tuple(o.cpu() for o in out)
            preds.append(MixtureHead.mean(cpu))
            draws.append(torch.stack([MixtureHead.sample(cpu, gen) for _ in range(n_samples)]))
        else:
            preds.append(out.cpu())
    targets = starts + cfg.window
    samples = torch.cat(draws, 1).numpy() if draws else None
    return Forecast(targets, series.truth[targets], torch.cat(preds).numpy(), samples)


@torch.no_grad()
def impute(model, series: Series, cfg: TrainConfig, split: str = "test") -> np.ndarray:
    """Causal imputation: walk forward in time, fill every hidden entry of x_t from
    p(x_t | imputed past, observed entries of x_t) and feed the filled value back.

    For the mixture this is exact conditioning: component responsibilities are
    updated with the entries observed at t, then the missing entries take the
    posterior mean. Returns the (T, N) filled series (scaled); only the split range
    is changed.
    """
    model.eval()
    device = torch.device(cfg.device)
    lo, hi = series.split_range(split)
    w = cfg.window
    filled = series.values.copy()
    for t in range(max(lo, w), hi):
        x = torch.from_numpy(filled[None, t - w:t])
        exo = torch.from_numpy(series.exo[None, t - w:t])
        obs = torch.from_numpy(series.mask[None, t - w:t])
        out = _params_last(model, x, exo, obs, cfg, device)
        hidden = ~series.mask[t]
        if not hidden.any():
            continue
        if _is_mixture(model):
            y = torch.from_numpy(np.nan_to_num(series.values[None, t])).to(device)
            ob = torch.from_numpy(series.mask[None, t]).to(device)
            est = model.head.conditional_mean(out, y, ob)[0].cpu().numpy()
        else:
            est = out[0].cpu().numpy()
        filled[t, hidden] = est[hidden]
    return filled


@torch.no_grad()
def generate(model, series: Series, cfg: TrainConfig, n_samples: int, length: int,
             burn_in: int = 50, seed: int = 0) -> np.ndarray:
    """Autoregressive sampling x_t ~ p(· | x_{t-w:t}, y_{t-w:t}) appended one step at a time.

    Seeds are real windows from the *training* split; the first `burn_in` generated
    steps are discarded so samples do not start on real data. Covariates follow
    the real calendar after each seed. Returns (n_samples, length, N), scaled.
    """
    if not _is_mixture(model):
        raise ValueError("generation needs a probabilistic (mixture) model")
    model.eval()
    device = torch.device(cfg.device)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w, steps = cfg.window, burn_in + length
    max_start = series.split_range("train")[1] - w
    horizon = len(series.values) - (w + steps)
    starts = rng.integers(0, max(1, min(max_start, horizon)), size=n_samples)
    cur = torch.from_numpy(gather_windows(series.values, starts, w))
    exo_all = torch.from_numpy(gather_windows(series.exo, starts, w + steps))
    obs = torch.ones(cur.shape, dtype=torch.bool)
    out = []
    for k in range(steps):
        params = _params_last(model, cur, exo_all[:, k:k + w], obs, cfg, device)
        nxt = MixtureHead.sample(tuple(p.cpu() for p in params), gen)
        out.append(nxt)
        cur = torch.cat([cur[:, 1:], nxt[:, None]], 1)
    return torch.stack(out, 1)[:, burn_in:].numpy()


def real_windows(series: Series, split: str, length: int, n: int, seed: int = 0) -> np.ndarray:
    """n stride-1 windows of `length` from `split` (fully observed truth where available)."""
    lo, hi = series.split_range(split)
    if hi - lo < length:
        raise ValueError(f"split '{split}' shorter than {length}")
    rng = np.random.default_rng(seed)
    starts = rng.choice(np.arange(lo, hi - length + 1), size=min(n, hi - lo - length + 1), replace=False)
    return gather_windows(series.values, np.sort(starts), length)


# --------------------------------------------------------------------------- imputation masks

def point_mask(shape: tuple[int, int], p: float, seed: int) -> np.ndarray:
    """Each entry hidden independently with probability p (GRIN 'point missing')."""
    return np.random.default_rng(seed).random(shape) < p


def block_mask(shape: tuple[int, int], seed: int, p_fault: float = 0.0015, p_noise: float = 0.05,
               min_len: int = 12, max_len: int = 48) -> np.ndarray:
    """GRIN 'block missing': sensor failures of random length plus sparse point noise."""
    rng = np.random.default_rng(seed)
    t, n = shape
    mask = rng.random(shape) < p_noise
    for i in range(n):
        for s in np.flatnonzero(rng.random(t) < p_fault):
            mask[s:s + rng.integers(min_len, max_len + 1), i] = True
    return mask
