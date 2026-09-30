"""Temporal graphs over the time steps of a window.

All graphs are *causal*: A[b, t, j] > 0 only for j < t, i.e. step t aggregates
messages from its past. For the natural visibility graph this is exact, not an
approximation: whether j and t see each other depends only on the samples
between them, so the graph of the prefix x[0:t+1] is the induced subgraph of the
window's graph (prefix property). One graph per window therefore serves every
position of a teacher-forced sequence without leaking the future.

Visibility graphs are computed with RustyGraph (exact arithmetic, parallel batch
kernel). `vector` uses the vector visibility graph of Ren & Jin (2019): for a < b
all vectors are projected on x_a and the natural visibility criterion is applied
to the projections.
"""

from __future__ import annotations

import numpy as np
import rustygraph as rg

KINDS = ("vg", "hvg", "chain", "complete", "none")
WEIGHTS = ("binary", "similarity")


def _edges_to_dense(edges: np.ndarray, offsets: np.ndarray, batch: int, w: int) -> np.ndarray:
    """(E, 2) edges (a < b) per window → (B, w, w) with A[b, later, earlier] = 1."""
    a = np.zeros((batch, w, w), np.float32)
    counts = np.diff(offsets)
    win = np.repeat(np.arange(batch), counts)
    a[win, edges[:, 1], edges[:, 0]] = 1.0
    return a


def visibility_adjacency(windows: np.ndarray, horizontal: bool = False) -> np.ndarray:
    """Causal (vector) visibility adjacency for (B, w, N) windows."""
    x = np.ascontiguousarray(windows, dtype=np.float64)
    b, w, n = x.shape
    if n == 1:
        fn = rg.horizontal_visibility_batch if horizontal else rg.natural_visibility_batch
        edges, offsets = fn(x[..., 0])
    else:
        fn = rg.horizontal_vector_visibility_batch if horizontal else rg.natural_vector_visibility_batch
        edges, offsets = fn(x)
    return _edges_to_dense(np.asarray(edges), np.asarray(offsets), b, w)


def temporal_adjacency(windows: np.ndarray, kind: str = "vg", weight: str = "binary") -> np.ndarray:
    """(B, w, N) windows → (B, w, w) causal adjacency, rows normalised to sum 1 (or 0).

    kind: 'vg' natural (vector) visibility, 'hvg' horizontal, 'chain' j = t-1 only,
    'complete' every j < t, 'none' no edges.
    weight: 'binary', or 'similarity' = exp(-‖x_t - x_j‖ / median distance), which
    makes *similar* steps exchange larger messages (the original code used the
    distance itself, i.e. the opposite).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown temporal graph '{kind}' {KINDS}")
    if weight not in WEIGHTS:
        raise ValueError(f"unknown edge weight '{weight}' {WEIGHTS}")
    if windows.ndim != 3:
        raise ValueError(f"windows must be (B, w, N), got {windows.shape}")
    b, w, _ = windows.shape
    if kind in ("vg", "hvg"):
        a = visibility_adjacency(windows, horizontal=kind == "hvg")
    elif kind == "chain":
        a = np.broadcast_to(np.eye(w, k=-1, dtype=np.float32), (b, w, w)).copy()
    elif kind == "complete":
        a = np.broadcast_to(np.tril(np.ones((w, w), np.float32), k=-1), (b, w, w)).copy()
    else:
        a = np.zeros((b, w, w), np.float32)
    if weight == "similarity":
        d = np.linalg.norm(windows[:, :, None, :] - windows[:, None, :, :], axis=-1)
        scale = np.median(d[a > 0]) if np.any(a > 0) else 1.0
        a = a * np.exp(-d / max(scale, 1e-8)).astype(np.float32)
    deg = a.sum(-1, keepdims=True)
    return np.divide(a, deg, out=np.zeros_like(a), where=deg > 0)


def random_adjacency_like(adj: np.ndarray, seed: int = 0) -> np.ndarray:
    """Control graph: same number of edges and weights as `adj`, placed at random (no self loops)."""
    rng = np.random.default_rng(seed)
    n = len(adj)
    off = ~np.eye(n, dtype=bool)
    weights = adj[off]
    out = np.zeros_like(adj)
    out[off] = rng.permutation(weights)
    return out


def row_normalize(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(-1, keepdims=True)
    return np.divide(adj, deg, out=np.zeros_like(adj), where=deg > 0)
