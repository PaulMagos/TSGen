"""External baselines: shapes and protocol (skipped when not installed)."""

import numpy as np
import pytest

from tsgen import data, external


def small_series():
    raw, cols = data.synthetic_raw(400)
    return data.build_series("toy", raw, np.zeros((400, 0), np.float32), cols)


def test_training_sequences_use_train_split_only():
    s = small_series()
    seqs = external.training_sequences(s, 20, seed=0, max_sequences=50)
    tr = s.split_range("train")[1]
    assert seqs.shape == (50, 20, 6)
    train_windows = {w.tobytes() for w in data.gather_windows(s.values, np.arange(tr - 19), 20)}
    assert all(w.tobytes() in train_windows for w in seqs)


def test_dgan_shapes():
    if not external.VENDOR.exists():
        pytest.skip("DGAN not vendored: run scripts/install_baselines.sh")
    fake = external.dgan_generate(small_series(), 20, 5, n=7, seed=0, device="cpu", epochs=1)
    assert fake.shape == (7, 20, 6) and np.isfinite(fake).all()


def test_par_shapes():
    pytest.importorskip("deepecho")
    fake, history = external.par_generate(small_series(), 20, n=3, seed=0, device="cpu", epochs=1)
    assert len(history) == 1
    assert fake.shape == (3, 20, 6) and np.isfinite(fake).all()
