#!/usr/bin/env bash
# External baselines for the corrected protocol, without touching the core versions:
#   PAR  (deepecho) and GRIN (torch-spatiotemporal) install cleanly into .venv;
#   DGAN (gretel-synthetics) would downgrade numpy/pandas/scipy and pull TensorFlow, but
#   its timeseries_dgan module is pure PyTorch: vendor it (+ category_encoders) with
#   --no-deps into vendor/, which tsgen/external.py puts on sys.path.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
uv pip install -p "$PY" "deepecho==0.8.1" "torch-spatiotemporal==0.9.5"
rm -rf vendor && uv pip install --no-deps --target vendor "gretel-synthetics==0.22.2" "category-encoders==2.8.1"
PYTHONPATH=vendor "$PY" -c "from gretel_synthetics.timeseries_dgan.dgan import DGAN; import deepecho, tsl; print('baselines ok')"
