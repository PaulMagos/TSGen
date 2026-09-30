"""Datasets, chronological splits and windowing.

Every dataset is returned as one `Series` over the full time axis plus split
boundaries. Scaling is fit on the training range only, so validation and test
never influence preprocessing. Windows are stride-1 and a window belongs to a
split when its *targets* lie in that split; its inputs may reach back into
earlier data (that is past information and legitimate at test time).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("TSGEN_DATA", Path(__file__).resolve().parent.parent / "data_cache"))
EXCHANGE_URL = ("https://github.com/TorchSpatiotemporal/multivariate-time-series-data/"
                "blob/master/exchange_rate/exchange_rate.txt.gz?raw=true")
AQI_URL = "https://drive.switch.ch/index.php/s/W0fRqotjHxIndPj/download"
SPLITS = (0.7, 0.1, 0.2)
EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Series:
    """A scaled multivariate series with masks, covariates and split bounds.

    values: (T, N) float32 model inputs, scaled to [0, 1] on the training range and
        NaN-free: entries the model may not see are forward-filled (causal).
    mask: (T, N) bool, True where the model may see the value (observed and not
        held out). Losses use it as the target mask.
    truth: (T, N) float32 scaled observations with NaN where never observed; the
        reference for every score (held-out entries included).
    eval_mask: (T, N) bool or None, observed entries hidden from the model in all
        splits and scored only for imputation (GRIN protocol for AQI-36).
    exo: (T, E) float32 exogenous covariates (E may be 0).
    adjacency: (N, N) float32 static spatial graph or None.
    bounds: (train_end, val_end); train = [0, train_end), val = [train_end,
        val_end), test = [val_end, T).
    """

    name: str
    values: np.ndarray
    mask: np.ndarray
    truth: np.ndarray
    exo: np.ndarray
    bounds: tuple[int, int]
    scale_min: np.ndarray
    scale_range: np.ndarray
    eval_mask: np.ndarray | None = None
    adjacency: np.ndarray | None = None
    columns: list[str] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return self.values.shape[1]

    @property
    def n_exo(self) -> int:
        return self.exo.shape[1]

    def split_range(self, split: str) -> tuple[int, int]:
        tr, va = self.bounds
        ranges = {"train": (0, tr), "val": (tr, va), "test": (va, len(self.values))}
        if split not in ranges:
            raise ValueError(f"unknown split '{split}' (train, val, test)")
        return ranges[split]

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale_range + self.scale_min


def chronological_bounds(n: int, fractions: tuple[float, float, float] = SPLITS) -> tuple[int, int]:
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError(f"split fractions must be positive and sum to 1, got {fractions}")
    train_end = int(round(n * fractions[0]))
    val_end = int(round(n * (fractions[0] + fractions[1])))
    return train_end, val_end


def fit_minmax(raw: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable min and range over the training rows, NaN-aware."""
    lo = np.nanmin(raw[:train_end], axis=0)
    rng = np.nanmax(raw[:train_end], axis=0) - lo
    return lo, np.where(rng > 0, rng, 1.0)


def _ffill(x: np.ndarray) -> np.ndarray:
    """Causal fill: carry the last observation forward; leading NaNs take the first observation."""
    df = pd.DataFrame(x).ffill().bfill()
    return df.to_numpy(dtype=np.float32)


def calendar_encoding(index: pd.DatetimeIndex, units: tuple[str, ...]) -> np.ndarray:
    """sin/cos of hour-of-day ('hour'), day-of-week ('dow') and day-of-year ('doy')."""
    periods = {
        "hour": (index.hour + index.minute / 60.0, 24.0),
        "dow": (index.dayofweek + index.hour / 24.0, 7.0),
        "doy": (index.dayofyear - 1 + index.hour / 24.0, 365.25),
    }
    cols = []
    for u in units:
        value, period = periods[u]
        angle = 2 * np.pi * np.asarray(value, dtype=np.float64) / period
        cols += [np.sin(angle), np.cos(angle)]
    return np.stack(cols, axis=1).astype(np.float32) if cols else np.zeros((len(index), 0), np.float32)


def build_series(name: str, raw: np.ndarray, exo: np.ndarray, columns: list[str],
                 eval_mask: np.ndarray | None = None, adjacency: np.ndarray | None = None,
                 fractions: tuple[float, float, float] = SPLITS) -> Series:
    """Scale on train, hide held-out entries, fill inputs causally."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or len(raw) != len(exo):
        raise ValueError(f"raw must be (T, N) aligned with exo, got {raw.shape} and {exo.shape}")
    bounds = chronological_bounds(len(raw), fractions)
    observed = ~np.isnan(raw)
    if eval_mask is not None:
        eval_mask = eval_mask.astype(bool) & observed
        raw_visible = np.where(eval_mask, np.nan, raw)
    else:
        raw_visible = raw
    lo, rng = fit_minmax(raw_visible, bounds[0])
    visible = ~np.isnan(raw_visible)
    truth = ((raw - lo) / rng).astype(np.float32)
    return Series(name=name, values=_ffill((raw_visible - lo) / rng), mask=visible, truth=truth,
                  exo=exo.astype(np.float32),
                  bounds=bounds, scale_min=lo.astype(np.float32), scale_range=rng.astype(np.float32),
                  eval_mask=eval_mask, adjacency=adjacency, columns=columns)


# --------------------------------------------------------------------------- datasets

def synthetic_raw(n_steps: int = 3276, step: float = 0.1) -> tuple[np.ndarray, list[str]]:
    """The thesis' synthetic MTS (GenerateSyntheticDataset.ipynb): 6 deterministic signals of x = k·step."""
    x = np.arange(n_steps) * step
    s, c = np.sin(x), np.cos(x)
    cols = {"sin": s, "cos": c, "sin_cos": s + c, "scos": np.sin(c), "csin": np.cos(s), "tan": np.tan(s + c)}
    return np.stack(list(cols.values()), axis=1), list(cols)


def load_synthetic(n_steps: int = 3276) -> Series:
    raw, cols = synthetic_raw(n_steps)
    return build_series("Synthetic", raw, np.zeros((len(raw), 0), np.float32), cols)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed public dataset URL
    return dest


def load_exchange() -> Series:
    """Lai et al. (2018) daily exchange rates of 8 countries, 7588 steps.

    The file has no timestamps (tsl invents a calendar-day index, while the data are
    business days), so no calendar covariate is meaningful: E = 0.
    """
    path = _download(EXCHANGE_URL, CACHE / "exchange_rate.txt.gz")
    raw = pd.read_csv(path, header=None, compression="gzip").to_numpy(np.float64)
    return build_series("Exchange", raw, np.zeros((len(raw), 0), np.float32),
                        [f"cur{i}" for i in range(raw.shape[1])])


def _aqi_files() -> Path:
    root = CACHE / "aqi"
    if not (root / "small36.h5").exists():
        import zipfile
        archive = _download(AQI_URL, root / "aqi.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        archive.unlink()
    return root


def haversine_km(lat_lon_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat_lon_deg[:, 0]), np.radians(lat_lon_deg[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gaussian_kernel_adjacency(dist: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """W_ij = exp(-(d_ij/θ)²), θ = std(d), zeroed below `threshold` and on the diagonal (Li et al., 2018)."""
    w = np.exp(-np.square(dist / dist.std()))
    w[w < threshold] = 0.0
    np.fill_diagonal(w, 0.0)
    return w.astype(np.float32)


def load_aqi() -> Series:
    """AQI-36 PM2.5 (Zheng et al., 2015), hourly, 36 Beijing stations, 8759 steps.

    NaNs stay missing (mask), `eval_mask` is the standard held-out imputation mask
    (Yi et al., 2016; used by GRIN), adjacency is the thresholded Gaussian kernel of
    station distances. Covariates: hour of day and day of week.
    """
    root = _aqi_files()
    pm25 = pd.DataFrame(pd.read_hdf(root / "small36.h5", "pm25"))
    eval_mask = pd.DataFrame(pd.read_hdf(root / "small36.h5", "eval_mask")).to_numpy(bool)
    stations = pd.DataFrame(pd.read_hdf(root / "full437.h5", "stations"))
    dist = haversine_km(stations[["latitude", "longitude"]].to_numpy()[:36])
    exo = calendar_encoding(pd.DatetimeIndex(pm25.index), ("hour", "dow"))
    return build_series("AirQuality", pm25.to_numpy(np.float64), exo, [str(c) for c in pm25.columns],
                        eval_mask=eval_mask, adjacency=gaussian_kernel_adjacency(dist))


LOADERS = {"synthetic": load_synthetic, "exchange": load_exchange, "airquality": load_aqi}


def load(name: str) -> Series:
    key = name.lower().replace("_", "")
    if key not in LOADERS:
        raise ValueError(f"unknown dataset '{name}' ({', '.join(LOADERS)})")
    return LOADERS[key]()


def correlation_adjacency(series: Series, k: int = 3) -> np.ndarray:
    """Static graph for datasets without geography: top-k |Pearson| neighbours on the training range."""
    tr = series.split_range("train")[1]
    corr = np.abs(np.corrcoef(series.values[:tr].T))
    corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0.0)
    keep = np.argsort(-corr, axis=1)[:, :k]
    w = np.zeros_like(corr)
    rows = np.arange(len(corr))[:, None]
    w[rows, keep] = corr[rows, keep]
    return np.maximum(w, w.T).astype(np.float32)


# --------------------------------------------------------------------------- windows

def window_starts(series: Series, split: str, window: int) -> np.ndarray:
    """Start indices s of stride-1 windows x[s:s+w] → targets x[s+1:s+w+1].

    train: the whole window and its targets lie in train. val/test: the *last*
    target x[s+w] lies in the split (inputs may come from earlier splits).
    """
    lo, hi = series.split_range(split)
    if split == "train":
        starts = np.arange(0, hi - window)
    else:
        starts = np.arange(max(lo - window, 0), hi - window)
    if len(starts) == 0:
        raise ValueError(f"split '{split}' too short for window {window}")
    return starts


def gather_windows(arr: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """(len(starts), length, ...) view-free gather of arr[s:s+length]."""
    idx = starts[:, None] + np.arange(length)[None, :]
    return arr[idx]


def with_eval_mask(series: Series, eval_mask: np.ndarray) -> Series:
    """Same data with `eval_mask` entries hidden from the model (scaler refit without them)."""
    if eval_mask.shape != series.truth.shape:
        raise ValueError(f"eval_mask shape {eval_mask.shape} != data shape {series.truth.shape}")
    raw = series.denormalize(series.truth.astype(np.float64))
    return build_series(series.name, raw, series.exo, series.columns,
                        eval_mask=eval_mask, adjacency=series.adjacency)
