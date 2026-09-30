#!/usr/bin/env bash
# Start MLflow + Postgres. Creates .env (secrets) once, refreshes host/origin lists for
# the current LAN IP, and writes client credentials:
#   client.env      this Mac          (set -a; . infra/mlflow/client.env; set +a)
#   client-lan.env  other LAN machines (copy it there, chmod 600)
set -euo pipefail
cd "$(dirname "$0")"
umask 077
PORT=${MLFLOW_PORT:-5050}
IP=${MLFLOW_LAN_IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)}
[[ -n $IP ]] || { echo "cannot detect LAN IP; set MLFLOW_LAN_IP" >&2; exit 1; }
touch .env
grep -q '^POSTGRES_PASSWORD=' .env || printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
grep -q '^MLFLOW_ADMIN_PASSWORD=' .env || printf 'MLFLOW_ADMIN_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
grep -q '^MLFLOW_FLASK_SERVER_SECRET_KEY=' .env || printf 'MLFLOW_FLASK_SERVER_SECRET_KEY=%s\n' "$(openssl rand -hex 32)" >> .env
ARC=mlflow.tsgen-mlflow.arcbox.local
HOSTS="localhost,localhost:*,127.0.0.1,127.0.0.1:*,$IP,$IP:*,$ARC,$ARC:*"
ORIGINS="http://localhost:$PORT,http://127.0.0.1:$PORT,http://$IP:$PORT,http://$ARC:5000"
grep -v '^MLFLOW_ALLOWED_HOSTS=\|^MLFLOW_CORS_ORIGINS=\|^MLFLOW_PORT=' .env > .env.tmp || true
printf 'MLFLOW_ALLOWED_HOSTS=%s\nMLFLOW_CORS_ORIGINS=%s\nMLFLOW_PORT=%s\n' "$HOSTS" "$ORIGINS" "$PORT" >> .env.tmp
mv .env.tmp .env
PASS=$(grep '^MLFLOW_ADMIN_PASSWORD=' .env | cut -d= -f2)
printf 'MLFLOW_TRACKING_URI=http://localhost:%s\nMLFLOW_TRACKING_USERNAME=admin\nMLFLOW_TRACKING_PASSWORD=%s\n' "$PORT" "$PASS" > client.env
printf 'MLFLOW_TRACKING_URI=http://%s:%s\nMLFLOW_TRACKING_USERNAME=admin\nMLFLOW_TRACKING_PASSWORD=%s\n' "$IP" "$PORT" "$PASS" > client-lan.env
docker compose up -d --build
for _ in $(seq 60); do
  curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "MLflow up: http://localhost:$PORT (LAN: http://$IP:$PORT, user admin, password in infra/mlflow/.env)"; exit 0; }
  sleep 2
done
echo "MLflow did not become healthy; see: docker compose logs mlflow" >&2
exit 1
