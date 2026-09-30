"""GRIN re-implementation: shapes, blindness to hidden values, learning."""

import dataclasses

import numpy as np
import torch

from tsgen import data, grin, inference


def masked_series(t=600, seed=0):
    rng = np.random.default_rng(seed)
    base = np.sin(np.arange(t)[:, None] * 0.2 + np.arange(4)[None, :])
    raw = base + 0.05 * rng.standard_normal((t, 4))
    s = data.build_series("toy", raw, np.zeros((t, 0), np.float32), list("abcd"),
                          adjacency=np.ones((4, 4), np.float32) - np.eye(4, dtype=np.float32))
    return data.with_eval_mask(s, inference.point_mask(s.truth.shape, 0.2, seed))


def test_forward_shapes():
    model = grin.GRIN(torch.ones(4, 4), n_exo=2, hidden=8, ff=8)
    x, m, u = torch.rand(3, 10, 4), torch.rand(3, 10, 4) > 0.3, torch.rand(3, 10, 2)
    y, parts = model(x, m, u)
    assert y.shape == (3, 10, 4) and len(parts) == 4 and all(p.shape == y.shape for p in parts)


def test_imputation_is_blind_to_hidden_values():
    s1 = masked_series()
    raw = s1.denormalize(s1.truth.astype(np.float64))
    alt = np.where(s1.eval_mask, raw + 50, raw)
    s2 = dataclasses.replace(s1, truth=((alt - s1.scale_min) / s1.scale_range).astype(np.float32))
    cfg = grin.GrinConfig(window=12, hidden=8, ff=8, epochs=1)
    model = grin.fit(s1, s1.adjacency, cfg)
    lo, hi = s1.split_range("test")
    assert np.allclose(grin.impute_range(model, s1, cfg, lo, hi), grin.impute_range(model, s2, cfg, lo, hi))


def test_training_beats_untrained():
    s = masked_series()
    short = grin.fit(s, s.adjacency, grin.GrinConfig(window=12, hidden=16, ff=16, epochs=1, patience=1))
    longer = grin.fit(s, s.adjacency, grin.GrinConfig(window=12, hidden=16, ff=16, epochs=15, patience=15))
    assert longer.best_val < short.best_val
