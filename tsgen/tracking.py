"""Optional MLflow tracking for experiment runs.

Active only when MLFLOW_TRACKING_URI is set and mlflow is importable; otherwise every
call is a no-op, so runs never depend on the server. Logged per run: flattened config
(params), train/val loss per epoch, final scores, system metrics (CPU, RAM, GPU when
available), and the result JSON as an artifact.

    python -m tsgen.tracking backfill results/     # log finished JSONs not yet in MLflow
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
EXPERIMENT_PREFIX = "tsgen"
SECTIONS = ("prediction", "imputation", "generation", "fit")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def flatten_metrics(result: dict, sections: tuple[str, ...] = SECTIONS) -> dict[str, float]:
    """{'prediction': {'mae': x}} → {'prediction/mae': x}; non-numeric leaves dropped."""
    flat: dict[str, float] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}/{k}", v)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix] = float(node)

    for section in sections:
        if section in result:
            walk(section, result[section])
    if "params" in result:
        flat["n_params"] = float(result["params"])
    return flat


def flatten_params(config: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten_params(v, key + "."))
        elif v is not None:
            out[key] = ",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mlflow():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; tracking disabled")
        return None
    return mlflow


class Tracker:
    """Context manager around one MLflow run; failed runs are marked FAILED."""

    def __init__(self, dataset: str, run_name: str, config: dict, tags: dict[str, str] | None = None):
        self.mlflow = _mlflow()
        self.dataset, self.run_name, self.config = dataset, run_name, config
        self.tags = {"host": socket.gethostname(), **({"git_commit": c} if (c := _git_commit()) else {}),
                     **(tags or {})}

    @property
    def enabled(self) -> bool:
        return self.mlflow is not None

    def __enter__(self) -> "Tracker":
        if self.enabled:
            self.mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{self.dataset}")
            system = os.environ.get("TSGEN_SYSTEM_METRICS", "1") != "0"
            self.mlflow.start_run(run_name=self.run_name, tags=self.tags, log_system_metrics=system)
            self.mlflow.log_params(flatten_params(self.config))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.enabled:
            self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        return False

    def log_epoch(self, record: dict) -> None:
        if self.enabled:
            self.mlflow.log_metrics({"train_loss": record["train"], "val_loss": record["val"]},
                                    step=record["epoch"])

    def log_result(self, result: dict, path: Path | None = None) -> None:
        if not self.enabled:
            return
        self.mlflow.log_metrics(flatten_metrics(result))
        if path is not None:
            self.mlflow.log_artifact(str(path))
            self.mlflow.set_tag("result_path", _result_key(path))


def _result_key(path: Path) -> str:
    """dataset/model/seedK.json — stable id used to avoid logging a result twice."""
    return "/".join(Path(path).parts[-3:])


def backfill(root: Path) -> int:
    """Log every results/<dataset>/<model>/seed*.json that MLflow does not have yet."""
    mlflow = _mlflow()
    if mlflow is None:
        raise SystemExit("set MLFLOW_TRACKING_URI (and install mlflow) to backfill")
    done = 0
    for path in sorted(root.glob("*/*/seed*.json")):
        dataset, model = path.parts[-3], path.parts[-2]
        key = _result_key(path)
        mlflow.set_experiment(f"{EXPERIMENT_PREFIX}/{dataset}")
        if len(mlflow.search_runs(filter_string=f"tags.result_path = '{key}'", max_results=1)):
            continue
        result = json.loads(path.read_text())
        tracker = Tracker(dataset, f"{model} {path.stem}", result.get("config", {}), {"backfilled": "true"})
        os.environ["TSGEN_SYSTEM_METRICS"] = "0"
        with tracker:
            for record in result.get("fit", {}).get("history", []):
                tracker.log_epoch(record)
            tracker.log_result(result, path)
        done += 1
    return done


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="log finished result JSONs")
    b.add_argument("root", nargs="?", default="results")
    args = ap.parse_args(argv)
    if args.cmd == "backfill":
        print(f"logged {backfill(Path(args.root))} runs")


if __name__ == "__main__":
    main()
