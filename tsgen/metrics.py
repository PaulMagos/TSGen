"""Evaluation metrics for forecasting / imputation and for time-series generation.

Conventions
-----------
* Point metrics take ``(y_true, y_pred)`` of equal shape (typically (T, N)) and an
  optional boolean ``mask`` of the same shape (True = score this entry).
* Generation metrics take ``(real, fake)`` windows of shape (S, L, N)
  (windows x length x variables); S may differ between the two, L and N may not.
* Argument order is always (truth, prediction) / (real, fake).
* Outputs are python floats or flat dicts of floats; everything random is driven
  by an explicit ``seed``.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.special import rel_entr
from sklearn.neighbors import NearestNeighbors
from torch import nn

__all__ = [
    "mae", "mse", "mase", "crps_samples", "marginal_metrics", "mmd_rbf", "acf_distance",
    "cross_corr_distance", "discriminative_score", "tstr_score", "memorisation",
    "vg_metrics", "generation_report",
]

_BATCH = 128
_LR = 5e-3
_TRAIN_FRAC = 0.8
_STD_FLOOR = 1e-8
_DEFAULT_MAX_LAG = 24

# --------------------------------------------------------------------------- validation


def _mask(mask: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    m = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"mask shape {m.shape} does not match data shape {shape}")
    if not m.any():
        raise ValueError("mask selects no entries (or data is empty)")
    return m


def _point_pair(y_true, y_pred, mask) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and return the masked 1-D (truth, prediction) entries."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"y_true shape {t.shape} != y_pred shape {p.shape}")
    m = _mask(mask, t.shape)
    return t[m], p[m]


def _windows(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"{name} must have shape (S, L, N), got {a.shape}")
    if min(a.shape) < 1:
        raise ValueError(f"{name} is empty: shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or inf")
    return a


def _gen_pair(real, fake, names: Tuple[str, str] = ("real", "fake")) -> Tuple[np.ndarray, np.ndarray]:
    r, f = _windows(real, names[0]), _windows(fake, names[1])
    if r.shape[1:] != f.shape[1:]:
        raise ValueError(f"(L, N) mismatch: {names[0]} {r.shape[1:]} vs {names[1]} {f.shape[1:]}")
    return r, f


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return at most ``n`` windows of ``x`` drawn without replacement (order kept)."""
    if len(x) <= n:
        return x
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]


def _flat(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


# --------------------------------------------------------------------------- point metrics


def mae(y_true, y_pred, mask=None) -> float:
    """Mean absolute error: mean(|y_true - y_pred|) over masked entries."""
    t, p = _point_pair(y_true, y_pred, mask)
    return float(np.mean(np.abs(t - p)))


def mse(y_true, y_pred, mask=None) -> float:
    """Mean squared error: mean((y_true - y_pred)^2) over masked entries."""
    t, p = _point_pair(y_true, y_pred, mask)
    return float(np.mean((t - p) ** 2))


def mase(y_true, y_pred, y_train, mask=None, m: int = 1) -> float:
    """Mean absolute scaled error (Hyndman & Koehler 2006).

    MASE = MAE(y_true, y_pred) / mean_{t,n} |y_train[t, n] - y_train[t - m, n]|,
    differences taken along the time axis (axis 0) of y_train (T_train, N), pooled over
    variables, NaN differences ignored. Persistence on a random walk gives ~1.
    """
    tr = np.asarray(y_train, dtype=np.float64)
    if tr.ndim != 2:
        raise ValueError(f"y_train must have shape (T_train, N), got {tr.shape}")
    if np.ndim(y_true) < 1 or np.shape(y_true)[-1] != tr.shape[1]:
        raise ValueError(f"y_true last axis {np.shape(y_true)} does not match y_train N={tr.shape[1]}")
    if not 1 <= m < tr.shape[0]:
        raise ValueError(f"seasonal period m must be in [1, {tr.shape[0] - 1}], got {m}")
    diffs = np.abs(tr[m:] - tr[:-m])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0 or diffs.mean() <= 0:
        raise ValueError("MASE scale is zero or undefined (constant or all-NaN y_train)")
    return mae(y_true, y_pred, mask) / float(diffs.mean())


def crps_samples(y_true, samples, mask=None) -> float:
    """Sample-based CRPS, averaged over masked entries (Gneiting & Raftery 2007).

    CRPS = E|X - y| - 1/2 E|X - X'| with X, X' drawn from the S samples. The second term
    uses the sorted-sample identity 1/2 E|X - X'| = (1/S^2) sum_i (2i - S - 1) x_(i).
    ``samples`` has shape (S, *y_true.shape). With S = 1 this equals the MAE.
    """
    y = np.asarray(y_true, dtype=np.float64)
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != y.ndim + 1 or x.shape[1:] != y.shape:
        raise ValueError(f"samples shape {x.shape} must be (S, *{y.shape})")
    s = x.shape[0]
    if s < 1:
        raise ValueError("samples must contain at least one sample")
    m = _mask(mask, y.shape)
    xs = np.sort(x[:, m], axis=0)  # (S, K)
    spread = ((2.0 * np.arange(1, s + 1) - s - 1) / s**2) @ xs
    return float(np.mean(np.abs(xs - y[m]).mean(axis=0) - spread))


# --------------------------------------------------------------------------- marginal / kernel


def _jsd_bits(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) of histograms on shared edges."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(a, edges)[0] / a.size
    q = np.histogram(b, edges)[0] / b.size
    mix = 0.5 * (p + q)
    jsd = 0.5 * (rel_entr(p, mix).sum() + rel_entr(q, mix).sum()) / np.log(2.0)
    return float(np.clip(jsd, 0.0, 1.0))


def marginal_metrics(real, fake, bins: int = 50) -> Dict[str, float]:
    """Per-variable marginal distances on pooled (S*L) values, averaged over variables.

    wasserstein: 1-D W1 distance; ks: two-sample Kolmogorov-Smirnov statistic;
    jsd: Jensen-Shannon divergence in bits with ``bins`` shared bins on the pooled range.
    These are blind to temporal order (see ``acf_distance``).
    """
    r, f = _gen_pair(real, fake)
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    out = {"wasserstein": [], "ks": [], "jsd": []}
    for n in range(r.shape[2]):
        a, b = r[:, :, n].ravel(), f[:, :, n].ravel()
        out["wasserstein"].append(stats.wasserstein_distance(a, b))
        out["ks"].append(stats.ks_2samp(a, b).statistic)
        out["jsd"].append(_jsd_bits(a, b, bins))
    return {k: float(np.mean(v)) for k, v in out.items()}


def mmd_rbf(real, fake, seed: int = 0, max_samples: int = 1000) -> float:
    """Unbiased MMD^2 with an RBF kernel on flattened windows (Gretton et al. 2012).

    k(x, y) = exp(-||x - y||^2 / (2 sigma^2)), sigma = median pairwise distance of the
    pooled sample (median heuristic). MMD^2_u = mean_{i!=j} k(x_i, x_j)
    + mean_{i!=j} k(y_i, y_j) - 2 mean_{i,j} k(x_i, y_j). Each side is subsampled
    to ``max_samples`` windows with ``seed``. Can be slightly negative.
    """
    r, f = _gen_pair(real, fake)
    rng = np.random.default_rng(seed)
    x, y = _flat(_subsample(r, max_samples, rng)), _flat(_subsample(f, max_samples, rng))
    if len(x) < 2 or len(y) < 2:
        raise ValueError("mmd_rbf needs at least 2 windows per side")
    z = np.vstack([x, y])
    d2 = np.maximum(cdist(z, z, "sqeuclidean"), 0.0)
    med = float(np.median(np.sqrt(d2[np.triu_indices(len(z), k=1)])))
    k = np.exp(-d2 / (2.0 * (med**2 if med > 0 else 1.0)))
    n, mm = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    yy = (kyy.sum() - np.trace(kyy)) / (mm * (mm - 1))
    return float(xx + yy - 2.0 * kxy.mean())


# --------------------------------------------------------------------------- temporal structure


def _mean_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """(max_lag, N) autocorrelation rho_k = sum_t c_t c_{t+k} / sum_t c_t^2, window-averaged."""
    c = x - x.mean(axis=1, keepdims=True)
    var = (c**2).sum(axis=1)
    ok = np.ptp(x, axis=1) > 0  # constant windows -> rho = 0
    denom = np.where(ok, var, 1.0)
    rows = [np.where(ok, (c[:, :-k] * c[:, k:]).sum(axis=1) / denom, 0.0).mean(axis=0) for k in range(1, max_lag + 1)]
    return np.stack(rows)


def acf_distance(real, fake, max_lag: Optional[int] = None) -> float:
    """Mean |ACF_real(k, n) - ACF_fake(k, n)| over lags k = 1..max_lag and variables n.

    ACF is computed per window and variable (mean-centred, variance-normalised, constant
    windows -> 0) and averaged over windows. Default max_lag = min(L - 1, 24).
    """
    r, f = _gen_pair(real, fake)
    length = r.shape[1]
    if length < 2:
        raise ValueError("acf_distance needs windows of length >= 2")
    k = min(length - 1, _DEFAULT_MAX_LAG) if max_lag is None else int(max_lag)
    if not 1 <= k <= length - 1:
        raise ValueError(f"max_lag must be in [1, {length - 1}], got {k}")
    return float(np.mean(np.abs(_mean_acf(r, k) - _mean_acf(f, k))))


def _unit(seg: np.ndarray) -> np.ndarray:
    """Centre each (window, variable) segment and scale to unit norm (constant -> 0)."""
    c = seg - seg.mean(axis=1, keepdims=True)
    norm = np.sqrt((c**2).sum(axis=1, keepdims=True))
    ok = np.ptp(seg, axis=1, keepdims=True) > 0
    return np.where(ok, c / np.where(ok, norm, 1.0), 0.0)


def _mean_cross_corr(x: np.ndarray, lag: int) -> np.ndarray:
    """(N, N) matrix C[i, j] = mean_s corr(x_s[t, i], x_s[t + lag, j])."""
    length = x.shape[1]
    a, b = _unit(x[:, : length - lag]), _unit(x[:, lag:])
    return np.einsum("sti,stj->ij", a, b) / len(x)


def cross_corr_distance(real, fake, lags: Sequence[int] = (0, 1)) -> float:
    """Mean |C_real - C_fake| of window-averaged lagged Pearson cross-correlation matrices.

    Lag 0 uses off-diagonal entries only (diagonal is trivially 1); lag > 0 uses all N^2
    entries. The per-lag values are averaged over ``lags``. N = 1 returns 0.0.
    """
    r, f = _gen_pair(real, fake)
    length, n_var = r.shape[1:]
    lags = tuple(int(k) for k in lags)
    if not lags or any(not 0 <= k <= length - 2 for k in lags):
        raise ValueError(f"lags must be a non-empty sequence in [0, {length - 2}], got {lags}")
    if n_var == 1:
        return 0.0
    off_diag = ~np.eye(n_var, dtype=bool)
    vals = []
    for k in lags:
        diff = np.abs(_mean_cross_corr(r, k) - _mean_cross_corr(f, k))
        vals.append(diff[off_diag].mean() if k == 0 else diff.mean())
    return float(np.mean(vals))


# --------------------------------------------------------------------------- learned metrics


class _GRUHead(nn.Module):
    """GRU encoder + linear head, applied to the last step or to every step."""

    def __init__(self, n_in: int, hidden: int, n_out: int, layers: int, last_only: bool) -> None:
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)
        self.last_only = last_only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h[:, -1] if self.last_only else h)


def _hidden(n_var: int) -> int:
    return max(n_var // 2, 4)  # TimeGAN convention: hidden = dim / 2


def _norm_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-variable mean / std over (S, L), std floored to avoid division by zero."""
    return x.mean(axis=(0, 1)), np.maximum(x.std(axis=(0, 1)), _STD_FLOOR)


def _tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)


def _fit(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: Callable, epochs: int, seed: int) -> None:
    """Adam minibatch training with a seeded shuffling generator."""
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=_LR)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=gen).to(x.device)
        for i in range(0, len(x), _BATCH):
            idx = perm[i : i + _BATCH]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()
    model.eval()


def discriminative_score(real, fake, seed: int = 0, epochs: int = 40, device: str = "cpu") -> float:
    """Post-hoc discriminative score (Yoon et al., TimeGAN, NeurIPS 2019).

    Balance by subsampling both sides to min(S_real, S_fake); train a 2-layer GRU
    classifier (hidden = max(N // 2, 4)) on a random 80 % split (inputs standardised
    with train-split statistics) and return |accuracy - 0.5| on the held-out 20 %.
    0 = indistinguishable, 0.5 = perfectly separable.
    """
    r, f = _gen_pair(real, fake)
    rng = np.random.default_rng(seed)
    n = min(len(r), len(f))
    if n < 5:
        raise ValueError("discriminative_score needs at least 5 windows per side")
    x = np.concatenate([_subsample(r, n, rng), _subsample(f, n, rng)])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    perm = rng.permutation(2 * n)
    n_tr = int(round(_TRAIN_FRAC * 2 * n))
    tr, te = perm[:n_tr], perm[n_tr:]
    mu, sd = _norm_stats(x[tr])
    xt, yt = _tensor((x - mu) / sd, device), _tensor(y, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _GRUHead(x.shape[2], _hidden(x.shape[2]), 1, layers=2, last_only=True).to(device)
        _fit(model, xt[tr], yt[tr].unsqueeze(-1), nn.BCEWithLogitsLoss(), epochs, seed)
        with torch.no_grad():
            pred = (model(xt[te]).squeeze(-1) > 0).float()
    acc = float(np.mean(pred.cpu().numpy() == y[te]))
    return abs(acc - 0.5)


def _forecast_mae(train: np.ndarray, test: np.ndarray, seed: int, epochs: int, device: str) -> float:
    """Train a 1-layer GRU one-step forecaster on ``train``; MAE on ``test`` (original units)."""
    mu, sd = _norm_stats(train)
    xtr, xte = _tensor((train - mu) / sd, device), _tensor((test - mu) / sd, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        n_var = train.shape[2]
        model = _GRUHead(n_var, _hidden(n_var), n_var, layers=1, last_only=False).to(device)
        _fit(model, xtr[:, :-1], xtr[:, 1:], nn.L1Loss(), epochs, seed)
        with torch.no_grad():
            pred = model(xte[:, :-1]).cpu().numpy().astype(np.float64)
    return float(np.mean(np.abs(pred * sd + mu - test[:, 1:])))


def tstr_score(real_train, fake, real_test, seed: int = 0, epochs: int = 40, device: str = "cpu") -> Dict[str, float]:
    """Predictive score / Train-on-Synthetic-Test-on-Real (Esteban et al. 2017; Yoon et al. 2019).

    A GRU predicts x_{t+1} (all variables) from x_{<=t}, trained with L1 loss on ``fake``
    (TSTR) and, identically, on ``real_train`` (TRTR); both are scored by MAE on
    ``real_test``. Each model standardises inputs with its own training-set statistics.
    Returns {'tstr_mae', 'trtr_mae', 'ratio' = tstr / trtr}.
    """
    tr, fk = _gen_pair(real_train, fake, ("real_train", "fake"))
    _, te = _gen_pair(tr, real_test, ("real_train", "real_test"))
    if tr.shape[1] < 2:
        raise ValueError("tstr_score needs windows of length >= 2")
    tstr = _forecast_mae(fk, te, seed, epochs, device)
    trtr = _forecast_mae(tr, te, seed, epochs, device)
    return {"tstr_mae": tstr, "trtr_mae": trtr, "ratio": tstr / trtr if trtr > 0 else float("inf")}


# --------------------------------------------------------------------------- memorisation / VG


def memorisation(train, fake, test, max_samples: int = 1000, seed: int = 0) -> Dict[str, float]:
    """Nearest-neighbour copy check on flattened windows (Euclidean).

    nn_fake_train = median_i min_j ||fake_i - train_j||, nn_test_train likewise for held-out
    real windows, ratio = nn_fake_train / nn_test_train (<< 1 suggests copying).
    The full ``train`` set is the reference; fake / test queries are subsampled to
    ``max_samples`` with ``seed``.
    """
    tr, fk = _gen_pair(train, fake, ("train", "fake"))
    _, te = _gen_pair(tr, test, ("train", "test"))
    rng = np.random.default_rng(seed)
    index = NearestNeighbors(n_neighbors=1).fit(_flat(tr))
    d_fake = float(np.median(index.kneighbors(_flat(_subsample(fk, max_samples, rng)))[0]))
    d_test = float(np.median(index.kneighbors(_flat(_subsample(te, max_samples, rng)))[0]))
    ratio = d_fake / d_test if d_test > 0 else float("nan")
    return {"nn_fake_train": d_fake, "nn_test_train": d_test, "ratio": ratio}


def vg_metrics(real, fake) -> Dict[str, float]:
    """Visibility-graph fidelity via ``rustygraph.metrics.vg_fidelity(real, fake)``.

    JSDs (bits) between NVG/HVG/VVG degree and motif distributions, absolute differences of
    scalar descriptors (irreversibility, temporal-structure index, multiplex), the mean JSD
    ``vg_divergence`` and a real-vs-real ``baseline_*`` split. Needs L >= 4.
    """
    from rustygraph.metrics import vg_fidelity

    r, f = _gen_pair(real, fake)
    res = vg_fidelity(np.ascontiguousarray(r), np.ascontiguousarray(f))
    return {k: float(v) for k, v in res.items() if isinstance(v, (int, float, np.integer, np.floating))}


# --------------------------------------------------------------------------- report


def generation_report(
    real, fake, real_train=None, real_test=None, seed: int = 0, device: str = "cpu"
) -> Dict[str, float]:
    """Run all generation metrics on equal-size real / fake sets and return a flat dict.

    The larger of real / fake is first subsampled (``seed``) to the smaller size.
    TSTR and memorisation are added only when both ``real_train`` and ``real_test`` are
    given (memorisation uses real_train as reference and real_test as the control).
    """
    r, f = _gen_pair(real, fake)
    rng = np.random.default_rng(seed)
    n = min(len(r), len(f))
    r, f = _subsample(r, n, rng), _subsample(f, n, rng)
    out: Dict[str, float] = dict(marginal_metrics(r, f))
    out["mmd_rbf"] = mmd_rbf(r, f, seed=seed)
    out["acf_distance"] = acf_distance(r, f)
    out["cross_corr_distance"] = cross_corr_distance(r, f)
    out["discriminative_score"] = discriminative_score(r, f, seed=seed, device=device)
    out.update({(k if k.startswith("vg_") else f"vg_{k}"): v for k, v in vg_metrics(r, f).items()})
    if real_train is not None and real_test is not None:
        tstr = tstr_score(real_train, f, real_test, seed=seed, device=device)
        out.update({"tstr_mae": tstr["tstr_mae"], "trtr_mae": tstr["trtr_mae"], "tstr_ratio": tstr["ratio"]})
        mem = memorisation(real_train, f, real_test, seed=seed)
        out.update({f"mem_{k}": v for k, v in mem.items()})
    return out
