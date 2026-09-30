#!/usr/bin/env bash
# Full experiment grid for the thesis re-run (GPU box). Resumable: skips finished JSONs.
#   SEEDS="0 1 2 3 4" EPOCHS=200 DEVICE=cuda bash scripts/run_grid.sh
# With MLFLOW_TRACKING_URI set, every run is also logged to MLflow (tsgen/tracking.py).
# A failing run is recorded in $OUT/failed.txt and the grid continues.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SEEDS=${SEEDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-200}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-results}
DATASETS=${DATASETS:-"synthetic exchange airquality"}

run() {  # dataset model seed tag [extra args...]
  local ds=$1 model=$2 seed=$3 tag=$4; shift 4
  local name
  name=$($PY -c "from tsgen.data import load; print(load('$ds').name)" 2>/dev/null)
  local f="$OUT/$name/$model$tag/seed$seed.json"
  [[ -f $f ]] && return 0
  echo ">> $ds $model$tag seed $seed"
  if ! $PY run_experiment.py --dataset "$ds" --model "$model" --seed "$seed" --epochs "$EPOCHS" \
      --device "$DEVICE" --out "$OUT" --tag="$tag" "$@"; then
    echo "$(date -u +%FT%TZ) $ds $model$tag seed$seed" >> "$OUT/failed.txt"
    echo "!! FAILED $ds $model$tag seed $seed"
  fi
}

for ds in $DATASETS; do
  for seed in $SEEDS; do
    # baselines (deterministic ones need one seed only, but are cheap)
    for m in persistence linear var locf interp; do run "$ds" "$m" "$seed" ""; done
    # main ladder, parameter-matched
    for m in mdn gtm sgtm asgtm lstm rnn; do run "$ds" "$m" "$seed" ""; done
    # ablations: temporal graph type (A3), random spatial graph (A4), absolute levels (A5)
    for g in chain complete hvg; do
      run "$ds" gtm "$seed" "-$g" --temporal-graph "$g" --tasks prediction,generation
    done
    run "$ds" gtm "$seed" "-simw" --edge-weight similarity --tasks prediction,generation
    run "$ds" sgtm "$seed" "-randgraph" --spatial-graph random --tasks prediction,generation
    run "$ds" asgtm "$seed" "-absolute" --absolute --tasks prediction,generation
  done
done
$PY aggregate.py --results "$OUT" --latex "$OUT/tables.tex" > "$OUT/tables.md"
echo "tables: $OUT/tables.md $OUT/tables.tex"
