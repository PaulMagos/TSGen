"""Tests for tsgen.metrics."""

import numpy as np
import pytest
from scipy.stats import norm

from tsgen import metrics as M

# --------------------------------------------------------------------------- data helpers


def _sines(s, length=24, n=3, seed=0, noise=0.05):
    """Sine windows with random frequency / phase per window and variable."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)[None, :, None]
    freq = rng.uniform(0.1, 0.3, (s, 1, n))
    phase = rng.uniform(0, 2 * np.pi, (s, 1, n))
    return np.sin(2 * np.pi * freq * t + phase) + noise * rng.standard_normal((s, length, n))


def _ar1(s, length=24, n=3, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((s, length, n))
    x[:, 0] = rng.standard_normal((s, n)) / np.sqrt(1 - phi**2)
    for t in range(1, length):
        x[:, t] = phi * x[:, t - 1] + rng.standard_normal((s, n))
    return x


def _time_shuffle(x, seed=0):
    """Permute each window independently in time: identical values, no temporal order."""
    rng = np.random.default_rng(seed)
    idx = rng.random(x.shape[:2]).argsort(axis=1)
    return np.take_along_axis(x, idx[:, :, None], axis=1)


# --------------------------------------------------------------------------- point metrics


def test_mae_mse_hand_values_and_mask():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = np.array([[2.0, 2.0], [1.0, 8.0]])  # errors 1, 0, 2, 4
    assert M.mae(y, p) == pytest.approx(7 / 4)
    assert M.mse(y, p) == pytest.approx(21 / 4)
    mask = np.array([[True, False], [False, True]])
    assert M.mae(y, p, mask) == pytest.approx(5 / 2)
    assert M.mse(y, p, mask) == pytest.approx(17 / 2)
    assert M.mae(p, y) == M.mae(y, p)


def test_mase_uses_time_axis_diffs():
    y_train = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])  # time diff 1, node diff 10
    y = np.array([[4.0, 14.0]])
    p = np.array([[6.0, 16.0]])  # MAE = 2
    assert M.mase(y, p, y_train) == pytest.approx(2.0)
    assert M.mase(y, p, y_train, m=2) == pytest.approx(1.0)
    with_nan = y_train.copy()
    with_nan[1, 0] = np.nan  # NaN diffs ignored, remaining diffs still 1
    assert M.mase(y, p, with_nan) == pytest.approx(2.0)


def test_mase_persistence_random_walk_and_perfect():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.standard_normal((3000, 5)), axis=0)
    y_train, y_test = series[:2000], series[2000:]
    persistence = series[1999:-1]
    assert M.mase(y_test, persistence, y_train) == pytest.approx(1.0, abs=0.05)
    assert M.mase(y_test, y_test, y_train) == 0.0


def test_crps_matches_gaussian_closed_form():
    rng = np.random.default_rng(0)
    t, n, s = 20, 5, 4000
    mu = rng.normal(0, 2, (t, n))
    sigma = rng.uniform(0.5, 2.0, (t, n))
    y = mu + sigma * rng.standard_normal((t, n))
    samples = mu + sigma * rng.standard_normal((s, t, n))
    z = (y - mu) / sigma
    closed = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    assert M.crps_samples(y, samples) == pytest.approx(closed.mean(), rel=0.02)
    mask = np.zeros((t, n), bool)
    mask[:5] = True
    assert M.crps_samples(y, samples, mask) == pytest.approx(closed[mask].mean(), rel=0.02)


def test_crps_point_mass_equals_mae():
    rng = np.random.default_rng(1)
    y, p = rng.standard_normal((10, 3)), rng.standard_normal((10, 3))
    assert M.crps_samples(y, p[None]) == pytest.approx(M.mae(y, p))
    assert M.crps_samples(y, np.repeat(p[None], 7, axis=0)) == pytest.approx(M.mae(y, p))


# --------------------------------------------------------------------------- generation metrics


def test_marginal_identical_bounded_and_sensitive():
    x = _ar1(200, seed=0)
    same = M.marginal_metrics(x, x.copy())
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in same.values())
    other = _ar1(200, seed=1)
    close = M.marginal_metrics(x, other)
    far = M.marginal_metrics(x, 3.0 + 0.3 * other)
    assert 0.0 <= close["jsd"] <= 1.0 and 0.0 <= far["jsd"] <= 1.0
    for key in ("wasserstein", "ks", "jsd"):
        assert far[key] > close[key] + 0.1
    assert far["jsd"] > 0.5


def test_mmd_same_vs_shifted():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((300, 12, 2)), rng.standard_normal((300, 12, 2))
    same = M.mmd_rbf(a, b)
    shifted = M.mmd_rbf(a, b + 1.0)
    assert abs(same) < 0.01
    assert shifted > 0.1
    assert M.mmd_rbf(a, b, seed=3, max_samples=100) == M.mmd_rbf(a, b, seed=3, max_samples=100)


def test_temporal_metrics_identical_zero():
    x = _ar1(100, n=4)
    assert M.acf_distance(x, x.copy()) == 0.0
    assert M.cross_corr_distance(x, x.copy(), lags=(0, 1, 3)) == pytest.approx(0.0, abs=1e-12)
    assert M.cross_corr_distance(x[:, :, :1], _ar1(50, n=1, seed=2)) == 0.0


def test_acf_constant_windows_are_zero():
    const = np.ones((10, 8, 2)) * 0.1  # float mean != 0.1 exactly: naive ACF would be garbage
    assert np.all(M._mean_acf(const, 7) == 0.0)
    x = _ar1(10, length=8, n=2)
    assert M.acf_distance(const, x) == pytest.approx(np.abs(M._mean_acf(x, 7)).mean())


def test_cross_corr_detects_coupling():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((200, 24, 1))
    coupled = np.concatenate([base, base + 0.1 * rng.standard_normal((200, 24, 1))], axis=2)
    indep = rng.standard_normal((200, 24, 2))
    assert M.cross_corr_distance(coupled, indep, lags=(0,)) > 0.8


def test_time_shuffle_fools_marginals_but_not_acf():
    """Thesis critique: marginal metrics cannot see temporal structure; ACF can."""
    real = _ar1(300, n=3, phi=0.9)
    shuffled = _time_shuffle(real)
    marg = M.marginal_metrics(real, shuffled)
    assert marg["wasserstein"] == pytest.approx(0.0, abs=1e-12)
    assert marg["ks"] == pytest.approx(0.0, abs=1e-12)
    assert marg["jsd"] == pytest.approx(0.0, abs=1e-12)
    shuffled_dist = M.acf_distance(real, shuffled)
    same_process_dist = M.acf_distance(real, _ar1(300, n=3, phi=0.9, seed=5))
    assert shuffled_dist > 0.1 and shuffled_dist > 10 * same_process_dist
    assert M.acf_distance(real, shuffled, max_lag=1) > 0.5


def test_discriminative_score():
    same = M.discriminative_score(_sines(500, seed=0), _sines(500, seed=1), seed=0)
    assert same < 0.15
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((500, 24, 3))
    assert M.discriminative_score(_sines(500, seed=0), noise, seed=0) > 0.35
    assert M.discriminative_score(_sines(60, seed=0), noise[:60], seed=4, epochs=2) == M.discriminative_score(
        _sines(60, seed=0), noise[:60], seed=4, epochs=2
    )


def test_tstr_score_keys_and_ordering():
    train, test = _ar1(200, n=3, seed=0), _ar1(100, n=3, seed=1)
    good = M.tstr_score(train, _ar1(200, n=3, seed=2), test, seed=0, epochs=20)
    bad = M.tstr_score(train, _time_shuffle(_ar1(200, n=3, seed=2)), test, seed=0, epochs=20)
    assert set(good) == {"tstr_mae", "trtr_mae", "ratio"}
    assert good["ratio"] == pytest.approx(good["tstr_mae"] / good["trtr_mae"])
    assert good["trtr_mae"] == bad["trtr_mae"]
    assert bad["tstr_mae"] > good["tstr_mae"]


def test_memorisation_detects_copies():
    train, test = _ar1(300, seed=0), _ar1(100, seed=1)
    copies = train[np.random.default_rng(0).choice(300, 150, replace=False)]
    res = M.memorisation(train, copies, test)
    assert res["nn_fake_train"] == pytest.approx(0.0, abs=1e-9)
    assert res["ratio"] < 0.01
    fresh = M.memorisation(train, _ar1(100, seed=2), test)
    assert 0.8 < fresh["ratio"] < 1.25


def test_vg_metrics_returns_floats():
    res = M.vg_metrics(_ar1(40, length=16, n=2), _ar1(40, length=16, n=2, seed=1).astype(np.float32))
    assert "vg_divergence" in res and all(isinstance(v, float) for v in res.values())


def test_generation_report_flat_and_optional_parts():
    real, fake = _ar1(80, length=12, n=2, seed=0), _ar1(120, length=12, n=2, seed=1)
    base = M.generation_report(real, fake)
    assert {"wasserstein", "ks", "jsd", "mmd_rbf", "acf_distance", "cross_corr_distance",
            "discriminative_score", "vg_divergence"} <= set(base)
    assert "tstr_mae" not in base and all(isinstance(v, float) for v in base.values())
    full = M.generation_report(real, fake, real_train=_ar1(80, length=12, n=2, seed=2),
                               real_test=_ar1(40, length=12, n=2, seed=3))
    assert {"tstr_mae", "trtr_mae", "tstr_ratio", "mem_ratio"} <= set(full)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "call",
    [
        lambda: M.mae(np.zeros((3, 2)), np.zeros((2, 3))),
        lambda: M.mse(np.zeros((3, 2)), np.zeros((3, 2)), mask=np.ones((3, 1), bool)),
        lambda: M.mae(np.zeros((3, 2)), np.zeros((3, 2)), mask=np.zeros((3, 2), bool)),
        lambda: M.mase(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((5, 3))),
        lambda: M.mase(np.zeros((3, 2)), np.zeros((3, 2)), np.ones((5, 2))),
        lambda: M.crps_samples(np.zeros((3, 2)), np.zeros((10, 2, 3))),
        lambda: M.marginal_metrics(np.zeros((4, 5, 2)), np.zeros((4, 5, 3))),
        lambda: M.mmd_rbf(np.zeros((4, 5)), np.zeros((4, 5))),
        lambda: M.acf_distance(np.zeros((4, 5, 2)), np.zeros((4, 6, 2))),
        lambda: M.acf_distance(np.zeros((4, 5, 2)), np.zeros((4, 5, 2)), max_lag=5),
        lambda: M.cross_corr_distance(np.zeros((4, 5, 2)), np.zeros((4, 5, 2)), lags=(4,)),
        lambda: M.discriminative_score(np.zeros((4, 5, 2)), np.full((4, 5, 2), np.nan)),
        lambda: M.memorisation(np.zeros((4, 5, 2)), np.zeros((4, 5, 2)), np.zeros((4, 5, 1))),
        lambda: M.tstr_score(np.zeros((4, 5, 2)), np.zeros((4, 5, 2)), np.zeros((4, 4, 2))),
    ],
)
def test_shape_validation_raises(call):
    with pytest.raises(ValueError):
        call()
