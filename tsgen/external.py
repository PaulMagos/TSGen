"""External generation baselines (DoppelGANger, PAR) on the corrected protocol.

Both learn from the same training windows as every other generator (length L, stride 1,
training split only, at most MAX_SEQUENCES chosen with the run seed) and return
(n, L, N) samples on the train-min-max scale, scored by the same metrics.

DGAN comes from gretel-synthetics' pure-PyTorch `timeseries_dgan`, vendored by
scripts/install_baselines.sh (the package's dependencies would downgrade the core
stack); PAR is DeepEcho's PARModel (Zhang et al., 2022).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .data import Series, gather_windows

log = logging.getLogger(__name__)
VENDOR = Path(__file__).resolve().parent.parent / "vendor"
MAX_SEQUENCES = 2000
DGAN_EPOCHS = 400      # library default
DGAN_BATCH = 256
PAR_EPOCHS = 100       # value of the thesis configuration (config/model/PAR.yaml)


def training_sequences(series: Series, length: int, seed: int, max_sequences: int = MAX_SEQUENCES) -> np.ndarray:
    """Stride-1 windows of `length` from the training split, at most `max_sequences` (seeded)."""
    tr = series.split_range("train")[1]
    starts = np.arange(0, tr - length + 1)
    if len(starts) > max_sequences:
        starts = np.sort(np.random.default_rng(seed).choice(starts, max_sequences, replace=False))
    return gather_windows(series.values, starts, length).astype(np.float32)


def _import_dgan():
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    try:
        from gretel_synthetics.timeseries_dgan.config import DGANConfig
        from gretel_synthetics.timeseries_dgan.dgan import DGAN
    except ImportError as exc:
        raise ImportError("DGAN not installed: run scripts/install_baselines.sh") from exc
    return DGAN, DGANConfig


def dgan_generate(series: Series, length: int, sample_len: int, n: int, seed: int, device: str,
                  epochs: int = DGAN_EPOCHS) -> np.ndarray:
    """DoppelGANger (Lin et al., IMC 2020): train on training windows, sample n sequences."""
    if length % sample_len:
        raise ValueError(f"DGAN needs sample_len dividing the sequence length ({sample_len} ∤ {length})")
    DGAN, DGANConfig = _import_dgan()
    torch.manual_seed(seed)
    np.random.seed(seed)
    features = training_sequences(series, length, seed)
    config = DGANConfig(max_sequence_len=length, sample_len=sample_len, batch_size=min(DGAN_BATCH, len(features)),
                        epochs=epochs, cuda=device.startswith("cuda"))
    model = DGAN(config)
    model.train_numpy(features=features)
    _, fake = model.generate_numpy(n)
    return np.asarray(fake, dtype=np.float32)


def par_generate(series: Series, length: int, n: int, seed: int, device: str, epochs: int = PAR_EPOCHS) -> np.ndarray:
    """PAR (DeepEcho): autoregressive RNN over per-column distribution parameters."""
    import pandas as pd
    from deepecho import PARModel

    torch.manual_seed(seed)
    np.random.seed(seed)
    seqs = training_sequences(series, length, seed)
    s, l, d = seqs.shape
    cols = [f"c{i}" for i in range(d)]
    frame = pd.DataFrame(seqs.reshape(s * l, d), columns=cols)
    frame.insert(0, "id", np.repeat(np.arange(s), l))
    model = PARModel(epochs=epochs, cuda=device.startswith("cuda"), verbose=False)
    model.fit(frame, entity_columns=["id"], data_types={c: "continuous" for c in cols})
    out = model.sample(num_entities=n, sequence_length=length)
    fake = out[cols].to_numpy(np.float32)
    return fake.reshape(n, length, d)
