"""FastPARModel must match deepecho's PARModel wherever it replaces a loop."""

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("deepecho")
from deepecho.models.par import PARModel  # noqa: E402

from tsgen.fast_par import FastPARModel  # noqa: E402

S, L, N = 12, 9, 3


def sequences(seed=0):
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.standard_normal((S, N, L)), axis=2)
    data[:, 2] = 5.0  # constant column: std = 0 path
    return [{"context": [], "data": [list(map(float, row)) for row in seq]} for seq in data]


def built(cls):
    m = cls(epochs=1, cuda=False, verbose=False)
    m._build(sequences(), [], ["continuous"] * N)
    return m


def test_tensor_building_matches_original():
    fast, orig = built(FastPARModel), built(PARModel)
    ref = torch.stack([orig._data_to_tensor(s["data"]) for s in sequences()], dim=1)
    assert torch.equal(fast._sequences_to_tensor(sequences()), ref)


def test_loss_matches_original():
    fast, orig = built(FastPARModel), built(PARModel)
    x = fast._sequences_to_tensor(sequences())
    y = torch.randn(x.shape)
    seq_len = torch.full((S,), x.shape[0], dtype=torch.long)
    a = fast._compute_loss(x[1:], y[:-1], seq_len)
    b = PARModel._compute_loss(orig, x[1:], y[:-1], seq_len)
    assert torch.allclose(a, b, rtol=1e-5)


def fitted(epochs=30):
    torch.manual_seed(0)
    frame = pd.DataFrame({"id": np.repeat(np.arange(S), L)})
    seqs = sequences()
    for c in range(N):
        frame[f"c{c}"] = np.concatenate([s["data"][c] for s in seqs])
    m = FastPARModel(epochs=epochs, cuda=False, verbose=False)
    m.fit(frame, entity_columns=["id"], data_types={f"c{c}": "continuous" for c in range(N)})
    return m


def test_incremental_state_equals_prefix_rerun():
    m = fitted(2)
    x = torch.randn(6, 1, m._data_dims)
    full = m._model(x, None)
    h, outs = None, []
    for t in range(6):
        o, h = m._model.rnn(m._model.down(x[t:t + 1]), h)
        outs.append(m._model.up(o))
    assert torch.allclose(torch.cat(outs), full, atol=1e-6)


def test_fit_records_loss_curve():
    m = fitted(5)
    assert list(m.loss_values["Epoch"]) == list(range(5)) and m.loss_values["Loss"].notna().all()


def test_batch_sampling_matches_original_sampler_in_distribution():
    m = fitted(40)
    g = torch.Generator().manual_seed(0)
    fast = m.sample_batch(400, L, generator=g)
    torch.manual_seed(1)
    orig = np.array([np.array(m.sample_sequence([], L), dtype=float).T for _ in range(400)])
    assert fast.shape == orig.shape == (400, L, N)
    assert np.allclose(fast[..., 2], 5.0) and np.allclose(orig[..., 2], 5.0)
    for c in (0, 1):
        se = np.sqrt(fast[..., c].var() / fast[..., c].size + orig[..., c].var() / orig[..., c].size) * np.sqrt(L)
        assert abs(fast[..., c].mean() - orig[..., c].mean()) < 4 * se
        assert abs(fast[..., c].std() / orig[..., c].std() - 1) < 0.15
