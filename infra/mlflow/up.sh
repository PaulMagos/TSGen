#!/usr/bin/env bash
# Start the tracking server; generate a local DB password once (kept out of git).
set -euo pipefail
cd "$(dirname "$0")"
URL=${MLFLOW_URL:-http://mlflow.tsgen-mlflow.arcbox.local:5000}
if [[ ! -f .env ]]; then
  umask 077
  printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" > .env
fi
docker compose up -d --build
for _ in $(seq 60); do
  curl -fsS $URL/health >/dev/null 2>&1 && { echo "MLflow up: $URL"; exit 0; }
  sleep 2
done
echo "MLflow did not become healthy; see: docker compose logs mlflow" >&2
exit 1
