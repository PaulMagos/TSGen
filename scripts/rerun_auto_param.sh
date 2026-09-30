#!/usr/bin/env bash
# Rerun AQI-36 and Synthetic with --param auto (levels; ADF: stationary) after
# scripts/supersede_relative.sh. Exchange keeps its relative runs (auto = relative).
#   DEVICE=cuda JOBS=4 bash scripts/rerun_auto_param.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export JOBS=${JOBS:-4} DEVICE=${DEVICE:-cuda}
S="synthetic airquality"
DATASETS="$S" MODELS="mdn gtm sgtm asgtm lstm rnn" TAG="" TASKS=prediction,generation,imputation \
  bash scripts/run_jobs.sh
for g in chain complete hvg; do
  DATASETS="$S" MODELS=gtm TAG="-$g" EXTRA="--temporal-graph $g" bash scripts/run_jobs.sh
done
DATASETS="$S" MODELS=gtm TAG=-simw EXTRA="--edge-weight similarity" bash scripts/run_jobs.sh
DATASETS="$S" MODELS=sgtm TAG=-randgraph EXTRA="--spatial-graph random" bash scripts/run_jobs.sh
DATASETS="$S" MODELS=asgtm TAG=-absolute EXTRA="--param absolute" bash scripts/run_jobs.sh
for h in 128 256; do
  DATASETS=airquality MODELS="mdn gtm asgtm" TAG="-h$h" EXTRA="--hidden $h" \
    TASKS=prediction,generation,imputation bash scripts/run_jobs.sh
done
