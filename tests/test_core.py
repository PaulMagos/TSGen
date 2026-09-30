"""Correctness tests for the fixed methodology core (each maps to a review finding)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch import distributions as D

from tsgen import baselines, data, graphs, inference, models, train

torch.set_default_dtype(torch.float32)


def toy_series(t=400, n=4, e=2, missing=0.0, seed=0) -> data.Series:
    rng = np.random.default_rng(seed)
    raw = np.cumsum(rng.standard_normal((t, n)), axis=0)
    if missing:
        raw[rng.random(raw.shape) < missing] = np.nan
    exo = rng.standard_normal((t, e)).astype(np.float32)
    adj = graphs.row_normalize(np.ones((n, n), np.float32) - np.eye(n, dtype=np.float32))
    return data.build_series("toy", raw, exo, [f"c{i}" for i in range(n)], adjacency=adj)


# ----------------------------------------------------------------------------- mixture head

def random_params(b=5, m=3, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    log_pi = torch.log_softmax(torch.randn(b, m, generator=g), -1)
    mu = torch.randn(b, m, n, generator=g)
    sigma = torch.rand(b, m, n, generator=g) + 0.2
    return log_pi, mu, sigma


def test_nll_equals_torch_mixture():
    """Review §1.1: the loss must be −log Σ_m π_m Π_i N(y_i | μ_mi, σ_mi²)."""
    params = random_params()
    y = torch.randn(5, 4)
    ref = D.MixtureSameFamily(D.Categorical(logits=params[0]),
                              D.Independent(D.Normal(params[1], params[2]), 1))
    head = models.MixtureHead(1, 4, 3)
    assert torch.allclose(head.nll(params, y), -ref.log_prob(y).mean(), atol=1e-5)


def test_missing_dims_are_marginalised():
    params = random_params()
    y = torch.randn(5, 4)
    obs = torch.tensor([True, False, True, False]).expand(5, 4)
    keep = [0, 2]
    ref = D.MixtureSameFamily(D.Categorical(logits=params[0]),
                              D.Independent(D.Normal(params[1][..., keep], params[2][..., keep]), 1))
    head = models.MixtureHead(1, 4, 3)
    assert torch.allclose(head.nll(params, y, obs), -ref.log_prob(y[:, keep]).mean(), atol=1e-5)


def test_sampling_is_component_sampling_not_weighted_sum():
    """Review §1.2: samples must be bimodal for a two-mode mixture (Σ π_m Y_m would be unimodal)."""
    log_pi = torch.log(torch.tensor([[0.5, 0.5]])).expand(20000, 2)
    mu = torch.tensor([[[-2.0], [2.0]]]).expand(20000, 2, 1)
    sigma = torch.full((20000, 2, 1), 0.1)
    x = models.MixtureHead.sample((log_pi, mu, sigma), torch.Generator().manual_seed(0))
    assert (x.abs() < 1).float().mean() < 0.01
    assert abs(float((x > 0).float().mean()) - 0.5) < 0.02
    assert abs(float(x.std()) - (4 + 0.01) ** 0.5) < 0.05


def test_mixture_recovers_two_modes():
    torch.manual_seed(0)
    head = models.MixtureHead(1, 1, 2)
    y = torch.cat([torch.randn(2000, 1) * 0.1 - 2, torch.randn(2000, 1) * 0.1 + 2])
    h = torch.ones(len(y), 1)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    with torch.no_grad():
        head.proj.bias[2:4] = torch.tensor([-1.0, 1.0])
    for _ in range(400):
        opt.zero_grad()
        head.nll(head(h), y).backward()
        opt.step()
    log_pi, mu, _ = head(h[:1])
    assert torch.allclose(log_pi.exp(), torch.tensor([[0.5, 0.5]]), atol=0.05)
    assert torch.allclose(mu.flatten().sort().values, torch.tensor([-2.0, 2.0]), atol=0.1)


# ----------------------------------------------------------------------------- causality

@pytest.mark.parametrize("variant", list(models.VARIANTS))
@pytest.mark.parametrize("graph", ["vg", "hvg", "complete"])
def test_outputs_before_t_ignore_the_future(variant, graph):
    """Review §1.4: perturbing x[t+1:] (and recomputing the graph) leaves outputs ≤ t unchanged."""
    torch.manual_seed(0)
    s = toy_series()
    model = models.build(models.ModelConfig(variant=variant, hidden=16), s.n_nodes, s.n_exo,
                         torch.from_numpy(s.adjacency)).eval()
    b = train.make_batch(s, np.array([10, 50, 90]), 20, graph if model.needs_temporal_graph else None)
    t = 12
    x2 = b.x.clone()
    x2[:, t + 1:] += torch.randn_like(x2[:, t + 1:]) * 3
    p2 = torch.from_numpy(graphs.temporal_adjacency(x2.numpy(), graph)) if model.needs_temporal_graph else None
    with torch.no_grad():
        out1 = model(b.x, b.exo, b.obs, b.p_time)
        out2 = model(x2, b.exo, b.obs, p2)
    for a, c in zip(out1, out2):
        assert torch.allclose(a[:, : t + 1], c[:, : t + 1], atol=1e-6)
        assert not torch.allclose(a[:, t + 1:], c[:, t + 1:])


def test_visibility_prefix_property():
    """The window VG restricted to a prefix equals the VG of the prefix."""
    rng = np.random.default_rng(1)
    x = rng.random((50, 24, 3))
    full = graphs.visibility_adjacency(x)
    for t in (5, 11, 23):
        assert np.array_equal(full[:, :t, :t], graphs.visibility_adjacency(x[:, :t]))


def test_temporal_graphs_are_strictly_causal():
    x = np.random.default_rng(0).random((4, 16, 3))
    for kind in graphs.KINDS:
        a = graphs.temporal_adjacency(x, kind)
        assert np.all(np.triu(a) == 0), kind


# ----------------------------------------------------------------------------- spatial block

def test_spatial_messages_follow_the_given_adjacency():
    """Review §1.7: node i only hears its graph neighbours (no offset by exogenous columns)."""
    torch.manual_seed(0)
    n = 5
    adj = torch.zeros(n, n)
    adj[0, 3] = 1.0  # 0 ← 3 only
    block = models.SpatialDiffusion(1, 4, hops=1)
    x = torch.randn(2, 3, n, 1)
    x2 = x.clone()
    x2[..., 3, :] += 5
    a1, a2 = block(x, adj), block(x2, adj)
    changed = (a1 - a2).abs().sum((0, 1, 3)) > 1e-6
    assert changed.tolist() == [True, False, False, True, False]


def test_adaptive_adjacency_is_sparse_and_stochastic():
    adj = models.AdaptiveAdjacency(8, 4, topk=3)()
    assert torch.all((adj > 0).sum(-1) <= 3)
    assert torch.allclose(adj.sum(-1), torch.ones(8))
    assert torch.all(adj.diagonal() == 0)


# ----------------------------------------------------------------------------- data protocol

def test_splits_are_chronological_and_scaler_uses_train_only():
    s = toy_series(t=1000)
    tr, va = s.bounds
    assert 0 < tr < va < 1000
    assert np.allclose(s.values[:tr].min(0), 0) and np.allclose(s.values[:tr].max(0), 1)
    raw = s.denormalize(s.truth.astype(np.float64))
    raw2 = raw.copy()
    raw2[va:] *= 100
    s2 = data.build_series("toy", raw2, s.exo, s.columns)
    assert np.allclose(s.values[:va], s2.values[:va])


def test_windows_do_not_cross_into_later_splits():
    s = toy_series(t=1000)
    w = 20
    tr, va = s.bounds
    train_starts = data.window_starts(s, "train", w)
    assert train_starts.max() + w < tr
    for split, (lo, hi) in (("val", (tr, va)), ("test", (va, 1000))):
        last_target = data.window_starts(s, split, w) + w
        assert last_target.min() == lo and last_target.max() == hi - 1


def test_imputation_never_reads_hidden_values():
    """Review §1.5: changing the true value of hidden entries must not change the imputation."""
    torch.manual_seed(0)
    s = toy_series(t=500)
    hidden = inference.point_mask(s.truth.shape, 0.3, seed=0)
    s1 = data.with_eval_mask(s, hidden)
    raw = s.denormalize(s.truth.astype(np.float64))
    raw_alt = np.where(hidden, raw + 50.0, raw)
    s2 = data.with_eval_mask(dataclasses.replace(s, truth=((raw_alt - s.scale_min) / s.scale_range).astype(np.float32)), hidden)
    cfg = train.TrainConfig(window=10, epochs=1)
    model = models.build(models.ModelConfig(variant="gtm", hidden=8, use_mask=True), 4, 2)
    train.fit(model, s1, cfg)
    f1, f2 = inference.impute(model, s1, cfg), inference.impute(model, s2, cfg)
    lo, hi = s1.split_range("test")
    assert np.allclose(s1.values, s2.values)  # inputs identical
    assert np.allclose(f1[lo:hi], f2[lo:hi])


# ----------------------------------------------------------------------------- training / baselines

def test_early_stopping_restores_best_weights():
    s = toy_series(t=600)
    cfg = train.TrainConfig(window=10, epochs=8, patience=2, lr=5e-2)
    model = models.build(models.ModelConfig(variant="mdn", hidden=8), 4, 2)
    hist = train.fit(model, s, cfg)
    val = train.last_step_only(train.split_batch(s, "val", cfg, False))
    assert np.isclose(train.evaluate_loss(model, val, "cpu"), hist["best_val"], rtol=1e-5)
    assert hist["best_val"] == min(h["val"] for h in hist["history"])


def test_var_recovers_var1():
    rng = np.random.default_rng(0)
    a = np.array([[0.8, 0.1], [0.0, 0.5]])
    x = np.zeros((6000, 2))
    for t in range(1, 6000):
        x[t] = a @ x[t - 1] + rng.standard_normal(2)
    s = data.build_series("var1", x, np.zeros((6000, 0), np.float32), ["x", "y"])
    var = baselines.VARBaseline(s, max_lags=1)
    assert np.allclose(var.coefs[0], a, atol=0.04)


def test_generation_shapes_and_burn_in():
    s = toy_series(t=400)
    cfg = train.TrainConfig(window=10, epochs=1)
    model = models.build(models.ModelConfig(variant="asgtm", hidden=8, topk=2), 4, 2)
    train.fit(model, s, cfg)
    g = inference.generate(model, s, cfg, n_samples=6, length=25, burn_in=7)
    assert g.shape == (6, 25, 4) and np.isfinite(g).all()
