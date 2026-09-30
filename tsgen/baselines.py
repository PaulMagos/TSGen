"""Parameter-free and linear sanity baselines. Any learned model must beat these."""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.api import VAR

from .data import Series

MAX_VAR_LAGS = 12


def _targets(series: Series, split: str, window: int) -> np.ndarray:
    lo, hi = series.split_range(split)
    return np.arange(max(lo, window), hi)


def persistence(series: Series, split: str = "test", window: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """x̂_t = x_{t-1}. Returns (targets, predictions)."""
    t = _targets(series, split, max(window, 1))
    return t, series.values[t - 1]


def linear_extrapolation(series: Series, split: str = "test", window: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """x̂_t = 2x_{t-1} − x_{t-2}."""
    t = _targets(series, split, max(window, 2))
    return t, 2 * series.values[t - 1] - series.values[t - 2]


class VARBaseline:
    """VAR(p) fit on the training range, p by AIC (≤ MAX_VAR_LAGS)."""

    def __init__(self, series: Series, max_lags: int = MAX_VAR_LAGS):
        train = series.values[: series.split_range("train")[1]].astype(np.float64)
        keep = train.std(0) > 1e-8
        self.keep = keep
        result = VAR(train[:, keep]).fit(maxlags=max_lags, ic="aic", trend="c")
        self.p = max(result.k_ar, 1)
        self.coefs = result.coefs if result.k_ar > 0 else np.zeros((1, keep.sum(), keep.sum()))
        self.intercept = result.intercept
        self.sigma_u = np.atleast_2d(result.sigma_u)
        self.series = series

    def _step(self, hist: np.ndarray) -> np.ndarray:
        """hist (..., p, K) oldest→newest → next (..., K)."""
        out = np.broadcast_to(self.intercept, hist.shape[:-2] + self.intercept.shape).copy()
        for i in range(self.p):
            out += hist[..., -1 - i, :] @ self.coefs[i].T
        return out

    def _full(self, reduced: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        full = fallback.copy()
        full[..., self.keep] = reduced
        return full

    def predict(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        t = _targets(self.series, split, self.p)
        x = self.series.values.astype(np.float64)
        hist = np.stack([x[t - self.p + i][:, self.keep] for i in range(self.p)], axis=1)
        return t, self._full(self._step(hist), x[t - 1]).astype(np.float32)

    def simulate(self, n_samples: int, length: int, burn_in: int = 50, seed: int = 0) -> np.ndarray:
        """Seeded by random training windows; Gaussian innovations N(0, Σ_u)."""
        rng = np.random.default_rng(seed)
        x = self.series.values.astype(np.float64)
        tr = self.series.split_range("train")[1]
        starts = rng.integers(0, tr - self.p, size=n_samples)
        hist = np.stack([x[starts + i][:, self.keep] for i in range(self.p)], axis=1)
        chol = np.linalg.cholesky(self.sigma_u + 1e-10 * np.eye(len(self.sigma_u)))
        const = x[starts + self.p - 1]
        out = []
        for _ in range(burn_in + length):
            nxt = self._step(hist) + rng.standard_normal((n_samples, len(chol))) @ chol.T
            out.append(self._full(nxt, const))
            hist = np.concatenate([hist[:, 1:], nxt[:, None]], 1)
        return np.stack(out, 1)[:, burn_in:].astype(np.float32)
