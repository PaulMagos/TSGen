"""Tracking must be a no-op without a server and must never break a run."""

from tsgen import tracking


def test_noop_without_uri(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with tracking.Tracker("toy", "run", {"a": 1}) as t:
        assert not t.enabled
        t.log_epoch({"epoch": 0, "train": 1.0, "val": 2.0})
        t.log_result({"prediction": {"mae": 0.1}})


def test_unreachable_server_disables_tracking(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # nothing listens here
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    monkeypatch.setenv("TSGEN_SYSTEM_METRICS", "0")
    with tracking.Tracker("toy", "run", {"a": 1}) as t:
        t.log_epoch({"epoch": 0, "train": 1.0, "val": 2.0})
        t.log_result({"prediction": {"mae": 0.1}})
    assert not t.enabled


def test_flatten():
    flat = tracking.flatten_metrics({"prediction": {"mae": 1.0}, "imputation": {"point": {"mse": 2}},
                                     "params": 10, "config": {"x": 1}})
    assert flat == {"prediction/mae": 1.0, "imputation/point/mse": 2.0, "n_params": 10.0}
    assert tracking.flatten_params({"a": {"b": [1, 2]}, "c": None}) == {"a.b": "1,2"}
