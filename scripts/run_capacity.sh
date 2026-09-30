#!/usr/bin/env bash
# Capacity sweep: does model size matter? --hidden sets the reference ASMR width that
# fixes the parameter budget of every model (main grid = 64; legacy AQI ASMR ≈ 0.72M
# params ≈ hidden 128-150 here). Resumable; JOBS runs share one GPU.
#   JOBS=3 DEVICE=cuda SEEDS="0 1 2 3 4" bash scripts/run_capacity.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PY=${PY:-.venv/bin/python} EPOCHS=${EPOCHS:-200} DEVICE=${DEVICE:-cuda} OUT=${OUT:-results}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-"airquality exchange"}
HIDDEN=${HIDDEN:-"128 256"}
MODELS=${MODELS:-"mdn mr asmr"}
JOBS=${JOBS:-3}

run_one() {  # dataset model hidden seed
  local ds=$1 m=$2 h=$3 s=$4 dir tasks
  case $ds in synthetic) dir=Synthetic ;; exchange) dir=Exchange ;; airquality) dir=AirQuality ;; esac
  [[ -f "$OUT/$dir/$m-h$h/seed$s.json" ]] && return 0
  # AQI imputation reuses the trained model; elsewhere it needs 2 extra trainings per run
  tasks=prediction,generation
  [[ $ds == airquality ]] && tasks=prediction,generation,imputation
  echo ">> $ds $m-h$h seed $s"
  if ! $PY run_experiment.py --dataset "$ds" --model "$m" --seed "$s" --epochs "$EPOCHS" \
      --device "$DEVICE" --out "$OUT" --hidden "$h" --tag="-h$h" --tasks "$tasks"; then
    echo "$(date -u +%FT%TZ) $ds $m-h$h seed$s" >> "$OUT/failed.txt"
    echo "!! FAILED $ds $m-h$h seed $s"
  fi
}
export -f run_one

for ds in $DATASETS; do for h in $HIDDEN; do for m in $MODELS; do for s in $SEEDS; do
  echo "$ds $m $h $s"
done; done; done; done | xargs -P "$JOBS" -L 1 bash -c 'run_one "$@"' _
