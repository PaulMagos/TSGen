#!/usr/bin/env bash
# Generic resumable sweep: DATASETS × MODELS × SEEDS with one TAG and EXTRA args.
#   TAG=-level EXTRA="--level-input" MODELS="mdn asgtm" JOBS=4 DEVICE=cuda bash scripts/run_jobs.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-.venv/bin/python} EPOCHS=${EPOCHS:-200} DEVICE=${DEVICE:-cuda} OUT=${OUT:-results}
export TAG=${TAG:?set TAG, e.g. -level} EXTRA=${EXTRA:-} TASKS=${TASKS:-prediction,generation}
DATASETS=${DATASETS:-"synthetic exchange airquality"}
MODELS=${MODELS:-"mdn gtm sgtm asgtm"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
JOBS=${JOBS:-4}

run_one() {  # dataset model seed
  local ds=$1 m=$2 s=$3 dir
  case $ds in synthetic) dir=Synthetic ;; exchange) dir=Exchange ;; airquality) dir=AirQuality ;; esac
  [[ -f "$OUT/$dir/$m$TAG/seed$s.json" ]] && return 0
  echo ">> $ds $m$TAG seed $s"
  # shellcheck disable=SC2086
  if ! $PY run_experiment.py --dataset "$ds" --model "$m" --seed "$s" --epochs "$EPOCHS" \
      --device "$DEVICE" --out "$OUT" --tag="$TAG" --tasks "$TASKS" $EXTRA; then
    echo "$(date -u +%FT%TZ) $ds $m$TAG seed$s" >> "$OUT/failed.txt"
    echo "!! FAILED $ds $m$TAG seed $s"
  fi
}
export -f run_one

for ds in $DATASETS; do for m in $MODELS; do for s in $SEEDS; do echo "$ds $m $s"; done; done; done \
  | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
