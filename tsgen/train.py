"""Window batches and the shared training loop (validation early stopping)."""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, replace

import numpy as np
import torch

from .data import Series, gather_windows, window_starts
from .graphs import temporal_adjacency

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor          # (B, w, N) inputs x[s:s+w]
    exo: torch.Tensor        # (B, w, E)
    obs: torch.Tensor        # (B, w, N) bool, input observed
    y: torch.Tensor          # (B, w, N) targets x[s+1:s+w+1]
    y_obs: torch.Tensor      # (B, w, N) bool, target observed (loss mask)
    p_time: torch.Tensor | None = None  # (B, w, w) causal temporal graph

    def to(self, device) -> "Batch":
        move = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()}
        return Batch(**move)

    def index(self, idx) -> "Batch":
        return Batch(**{k: (v[idx] if isinstance(v, torch.Tensor) else v) for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return len(self.x)


@dataclass(frozen=True)
class TrainConfig:
    window: int = 20
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 15
    grad_clip: float = 1.0
    temporal_graph: str = "vg"
    edge_weight: str = "binary"
    seed: int = 0
    device: str = "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batch(series: Series, starts: np.ndarray, window: int, graph: str | None = None,
               edge_weight: str = "binary") -> Batch:
    """Stride-1 windows starting at `starts`; `graph` = temporal graph kind or None."""
    w1 = gather_windows(series.values, starts, window + 1)
    m1 = gather_windows(series.mask, starts, window + 1)
    exo = gather_windows(series.exo, starts, window)
    x = w1[:, :-1]
    p = torch.from_numpy(temporal_adjacency(x, graph, edge_weight)) if graph else None
    return Batch(x=torch.from_numpy(x), exo=torch.from_numpy(exo), obs=torch.from_numpy(m1[:, :-1]),
                 y=torch.from_numpy(w1[:, 1:]), y_obs=torch.from_numpy(m1[:, 1:]), p_time=p)


def split_batch(series: Series, split: str, cfg: TrainConfig, needs_graph: bool) -> Batch:
    starts = window_starts(series, split, cfg.window)
    return make_batch(series, starts, cfg.window, cfg.temporal_graph if needs_graph else None, cfg.edge_weight)


def last_step_only(batch: Batch) -> Batch:
    """Score validation/test windows on their last target only, so every target counts once."""
    y_obs = torch.zeros_like(batch.y_obs)
    y_obs[:, -1] = batch.y_obs[:, -1]
    return replace(batch, y_obs=y_obs)


@torch.no_grad()
def evaluate_loss(model, batch: Batch, device, batch_size: int = 512) -> float:
    model.eval()
    total, weight = 0.0, 0.0
    for i in range(0, len(batch), batch_size):
        b = batch.index(slice(i, i + batch_size)).to(device)
        n = float(b.y_obs.any(-1).sum())
        total += model.loss(b).item() * n
        weight += n
    return total / max(weight, 1.0)


def fit(model: torch.nn.Module, series: Series, cfg: TrainConfig, on_epoch=None) -> dict:
    """Adam + early stopping on validation loss; restores the best weights (deep copy).

    `on_epoch(record)` is called after every epoch with {'epoch', 'train', 'val'}.
    """
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    needs_graph = getattr(model, "needs_temporal_graph", False)
    train = split_batch(series, "train", cfg, needs_graph)
    val = last_step_only(split_batch(series, "val", cfg, needs_graph))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(cfg.seed)
    best_loss, best_state, bad, history = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(train), generator=gen)
        running = 0.0
        for i in range(0, len(train), cfg.batch_size):
            b = train.index(perm[i:i + cfg.batch_size]).to(device)
            loss = model.loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * len(b)
        val_loss = evaluate_loss(model, val, device)
        history.append({"epoch": epoch, "train": running / len(train), "val": val_loss})
        if on_epoch is not None:
            on_epoch(history[-1])
        if val_loss < best_loss - 1e-6:
            best_loss, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        log.info("epoch %d train %.5f val %.5f", epoch, running / len(train), val_loss)
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return {"best_val": best_loss, "epochs_run": len(history), "history": history}
