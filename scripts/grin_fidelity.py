"""Fidelity check of tsgen/grin.py on the protocol of the GRIN paper (AQI-36).

Test months 3, 6, 9, 12; training on windows fully inside the other months (last 10 %
as validation); scored on the standard eval_mask inside the test months, MAE in µg/m³.
The paper reports MAE 10.51 ± 0.28 for GRIN (BRITS 14.50, MICE 30.37, mean 53.48).

    .venv/bin/python scripts/grin_fidelity.py --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tsgen import data, grin  # noqa: E402

TEST_MONTHS = (3, 6, 9, 12)


def window_starts(ok: np.ndarray, w: int) -> np.ndarray:
    """Starts of windows whose steps all satisfy `ok`."""
    run = np.convolve(ok.astype(int), np.ones(w, int), "valid")
    return np.flatnonzero(run == w)


def tensors(s: data.Series, starts: np.ndarray, w: int, device):
    t = [torch.from_numpy(data.gather_windows(a, starts, w)).to(device) for a in (s.values, s.mask, s.exo)]
    return t


@torch.no_grad()
def predict(model, s, starts, w, device, batch=256):
    acc, cnt = np.zeros(s.values.shape), np.zeros(s.values.shape)
    model.eval()
    for i in range(0, len(starts), batch):
        st = starts[i:i + batch]
        x, m, u = tensors(s, st, w, device)
        y, _ = model(x, m, u)
        idx = st[:, None] + np.arange(w)[None, :]
        np.add.at(acc, idx, y.cpu().numpy())
        np.add.at(cnt, idx, 1)
    return acc / np.maximum(cnt, 1), cnt > 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default="results/grin_fidelity")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    cfg = grin.GrinConfig(seed=args.seed, device=args.device, epochs=args.epochs)
    s = data.load_aqi()
    index = pd.DatetimeIndex(pd.read_hdf(data.CACHE / "aqi" / "small36.h5", "pm25").index)
    test = np.isin(index.month, TEST_MONTHS)
    w, device = cfg.window, torch.device(args.device)
    train_starts = window_starts(~test, w)
    n_val = len(train_starts) // 10
    fit_starts, val_starts = train_starts[:-n_val], train_starts[-n_val:]
    test_starts = window_starts(test, w)
    model = grin.GRIN(torch.from_numpy(s.adjacency), s.n_exo, cfg.hidden, cfg.ff, cfg.order).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    x_all, m_all, u_all = tensors(s, fit_starts, w, "cpu")
    truth = np.nan_to_num(s.truth)
    best, state, bad = np.inf, None, 0
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(x_all), generator=gen)
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            x, m, u = x_all[idx].to(device), m_all[idx].to(device), u_all[idx].to(device)
            keep = (torch.rand(m.shape, generator=gen) >= cfg.whiten).to(device)
            y, parts = model(x, m & keep, u)
            loss = grin._masked_mae(y, x, m) + sum(grin._masked_mae(p, x, m) for p in parts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pred, cov = predict(model, s, val_starts, w, device)
        vmask = s.eval_mask & cov
        val = float(np.abs(pred - truth)[vmask].mean())
        if val < best - 1e-6:
            best, state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(state)
    pred, cov = predict(model, s, test_starts, w, device)
    emask = s.eval_mask & test[:, None] & cov
    err = (pred - truth) * s.scale_range  # original units
    res = {"seed": args.seed, "epochs_run": epoch + 1, "val_mae_scaled": best,
           "test_mae": float(np.abs(err)[emask].mean()), "test_mse": float((err ** 2)[emask].mean()),
           "n_scored": int(emask.sum()), "paper_grin_mae": 10.51}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"seed{args.seed}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
