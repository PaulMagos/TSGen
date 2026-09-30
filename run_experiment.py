"""One experiment = dataset × model × seed → results/<dataset>/<model>/seed<k>.json.

    python run_experiment.py --dataset airquality --model asgtm --seed 0
    python run_experiment.py --dataset exchange --model var --seed 0 --tasks prediction,generation

Models: mdn gtm sgtm asgtm (mixture generators, parameter-matched to asgtm),
lstm rnn (point forecasters), persistence linear var locf interp (baselines).
Scores are on the train-min-max scale (the thesis' normalised scale).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsgen import baselines, data, graphs, inference, metrics, models, train

GENERATORS = tuple(models.VARIANTS)
FORECASTERS = ("lstm", "rnn")
PREDICTION_BASELINES = ("persistence", "linear", "var")
IMPUTATION_BASELINES = ("locf", "interp")
ALL_MODELS = GENERATORS + FORECASTERS + PREDICTION_BASELINES + IMPUTATION_BASELINES
TASKS = ("prediction", "imputation", "generation")
N_SAMPLES_CRPS = 100
N_GENERATED = 500
BURN_IN = 50
POINT_P = 0.25


@dataclass(frozen=True)
class DatasetSpec:
    window: int
    mixtures: int
    embedding: int
    gen_length: int


SPECS = {
    "Synthetic": DatasetSpec(window=15, mixtures=6, embedding=3, gen_length=63),
    "Exchange": DatasetSpec(window=20, mixtures=16, embedding=4, gen_length=216),
    "AirQuality": DatasetSpec(window=23, mixtures=36, embedding=20, gen_length=168),
}


def scored(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ok = ~np.isnan(y_true)
    return ok if mask is None else ok & mask


def point_scores(series: data.Series, targets, y_pred, samples=None) -> dict:
    y_true = series.truth[targets]
    ok = scored(y_true, y_pred)
    y_train = series.truth[: series.split_range("train")[1]]
    out = {"mae": metrics.mae(np.nan_to_num(y_true), y_pred, ok),
           "mse": metrics.mse(np.nan_to_num(y_true), y_pred, ok),
           "mase": metrics.mase(np.nan_to_num(y_true), y_pred, y_train, ok)}
    if samples is not None:
        out["crps"] = metrics.crps_samples(np.nan_to_num(y_true), samples, ok)
    return out


def imputation_scores(series: data.Series, filled: np.ndarray) -> dict:
    lo, hi = series.split_range("test")
    hidden = np.zeros_like(series.eval_mask)
    hidden[lo:hi] = series.eval_mask[lo:hi]
    truth = np.nan_to_num(series.truth)
    return {"mae": metrics.mae(truth, filled, hidden), "mse": metrics.mse(truth, filled, hidden),
            "n_scored": int(hidden.sum())}


def imputation_problems(series: data.Series, seed: int) -> dict[str, data.Series]:
    """AQI: the standard eval_mask. Others: synthetic point / block masks (seeded)."""
    if series.eval_mask is not None:
        return {"eval_mask": series}
    shape = series.truth.shape
    return {"point": data.with_eval_mask(series, inference.point_mask(shape, POINT_P, seed)),
            "block": data.with_eval_mask(series, inference.block_mask(shape, seed))}


SPATIAL_GRAPH = {"mode": "given", "seed": 0}  # set once from the CLI in main()


def static_adjacency(series: data.Series) -> torch.Tensor:
    """Geographic graph (AQI) or train-correlation graph; 'random' = same edges shuffled (control)."""
    adj = series.adjacency if series.adjacency is not None else data.correlation_adjacency(series)
    if SPATIAL_GRAPH["mode"] == "random":
        adj = graphs.random_adjacency_like(adj, SPATIAL_GRAPH["seed"])
    return torch.from_numpy(adj)


def make_model(name: str, series: data.Series, spec: DatasetSpec, hidden: int, relative: bool = True):
    use_mask = not series.mask.all()
    if name in FORECASTERS:
        return _matched_forecaster(name, series, use_mask, _budget(series, spec, hidden), relative)
    cfg = models.ModelConfig(variant=name, hidden=hidden, mixtures=spec.mixtures,
                             embedding=spec.embedding, use_mask=use_mask, relative=relative)
    return models.build(cfg, series.n_nodes, series.n_exo, static_adjacency(series),
                        param_budget=_budget(series, spec, hidden))


def _budget(series, spec, hidden) -> int:
    """Every learned model gets the size of ASGTM with the reference width."""
    cfg = models.ModelConfig(variant="asgtm", hidden=hidden, mixtures=spec.mixtures,
                             embedding=spec.embedding, use_mask=not series.mask.all())
    return models.count_params(models.build(cfg, series.n_nodes, series.n_exo, static_adjacency(series)))


def _matched_forecaster(cell, series, use_mask, budget, relative):
    sizes = range(8, 1025, 4)
    best = min(sizes, key=lambda h: abs(models.count_params(
        models.PointForecaster(cell, series.n_nodes, series.n_exo, h, use_mask)) - budget))
    return models.PointForecaster(cell, series.n_nodes, series.n_exo, best, use_mask, relative)


def run_learned(args, series, spec, tcfg) -> dict:
    out: dict = {}
    model = make_model(args.model, series, spec, args.hidden, not args.absolute)
    out["params"] = models.count_params(model)
    t0 = time.time()
    fit = train.fit(model, series, tcfg)
    out["fit"] = {"best_val": fit["best_val"], "epochs_run": fit["epochs_run"], "seconds": time.time() - t0}
    if "prediction" in args.tasks:
        f = inference.predict(model, series, tcfg, n_samples=N_SAMPLES_CRPS, seed=args.seed)
        out["prediction"] = point_scores(series, f.targets, f.y_pred, f.samples)
    if "generation" in args.tasks and args.model in GENERATORS:
        fake = inference.generate(model, series, tcfg, N_GENERATED, spec.gen_length, BURN_IN, args.seed)
        out["generation"] = generation_scores(series, fake, spec, args)
    if "imputation" in args.tasks:
        out["imputation"] = {}
        for label, problem in imputation_problems(series, args.seed).items():
            m = model
            if problem is not series:  # hidden entries must not be seen in training either
                m = make_model(args.model, problem, spec, args.hidden, not args.absolute)
                train.fit(m, problem, tcfg)
            out["imputation"][label] = imputation_scores(problem, inference.impute(m, problem, tcfg))
    return out


def generation_scores(series, fake, spec, args) -> dict:
    """Fidelity vs the training distribution (what the model learns; memorisation is checked
    against it) and vs the held-out test period (generalisation; includes any drift)."""
    real_test = inference.real_windows(series, "test", spec.gen_length, N_GENERATED, args.seed)
    real_train = inference.real_windows(series, "train", spec.gen_length, N_GENERATED, args.seed)
    return {"vs_train": metrics.generation_report(real_train, fake, real_train=real_train, real_test=real_test,
                                                  seed=args.seed, device=args.device),
            "vs_test": metrics.generation_report(real_test, fake, seed=args.seed, device=args.device)}


def run_baseline(args, series, spec) -> dict:
    out: dict = {}
    w = spec.window
    if args.model in PREDICTION_BASELINES and "prediction" in args.tasks:
        if args.model == "var":
            var = baselines.VARBaseline(series)
            t, p = var.predict()
            out["var_lags"] = var.p
            if "generation" in args.tasks:
                fake = var.simulate(N_GENERATED, spec.gen_length, BURN_IN, args.seed)
                out["generation"] = generation_scores(series, fake, spec, args)
        else:
            fn = baselines.persistence if args.model == "persistence" else baselines.linear_extrapolation
            t, p = fn(series, "test", w)
        keep = t >= series.split_range("test")[0] + 0
        out["prediction"] = point_scores(series, t[keep], p[keep])
    if args.model in IMPUTATION_BASELINES and "imputation" in args.tasks:
        out["imputation"] = {}
        for label, problem in imputation_problems(series, args.seed).items():
            if args.model == "locf":
                filled = problem.values
            else:  # linear interpolation over time — uses the future, an upper reference
                hidden = pd.DataFrame(np.where(problem.mask, problem.truth, np.nan))
                filled = hidden.interpolate(limit_direction="both").to_numpy(np.float32)
            out["imputation"][label] = imputation_scores(problem, filled)
    return out


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(data.LOADERS))
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64, help="reference ASGTM width for the parameter budget")
    ap.add_argument("--temporal-graph", default="vg", choices=["vg", "hvg", "chain", "complete", "none"])
    ap.add_argument("--edge-weight", default="binary", choices=["binary", "similarity"])
    ap.add_argument("--absolute", action="store_true",
                    help="ablation: absolute inputs/means (the original model) instead of level-relative")
    ap.add_argument("--spatial-graph", default="given", choices=["given", "random"],
                    help="ablation: replace the static graph by a random one with the same weights")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="", help="suffix for ablation runs, e.g. '-chain'")
    args = ap.parse_args(argv)
    args.tasks = tuple(t for t in args.tasks.split(",") if t)
    if bad := set(args.tasks) - set(TASKS):
        ap.error(f"unknown tasks {bad}")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    train.seed_everything(args.seed)
    SPATIAL_GRAPH.update(mode=args.spatial_graph, seed=args.seed)
    series = data.load(args.dataset)
    spec = SPECS[series.name]
    tcfg = train.TrainConfig(window=spec.window, epochs=args.epochs, seed=args.seed, device=args.device,
                             temporal_graph=args.temporal_graph, edge_weight=args.edge_weight)
    learned = args.model in GENERATORS + FORECASTERS
    result = run_learned(args, series, spec, tcfg) if learned else run_baseline(args, series, spec)
    result["config"] = {**{k: v for k, v in vars(args).items() if k != "tasks"}, "tasks": list(args.tasks),
                        "spec": asdict(spec), "train": asdict(tcfg)}
    path = Path(args.out) / series.name / f"{args.model}{args.tag}" / f"seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=float))
    return path


if __name__ == "__main__":
    print(main())
