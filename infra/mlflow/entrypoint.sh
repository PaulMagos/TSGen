#!/bin/sh
# Write the basic-auth config from the environment, then start the server.
# admin_password only seeds the auth DB on first start; change it later in the UI/API.
set -eu
: "${POSTGRES_PASSWORD:?}" "${MLFLOW_ADMIN_PASSWORD:?}" "${MLFLOW_FLASK_SERVER_SECRET_KEY:?}"
mkdir -p /mlauth
umask 077
cat > /mlauth/basic_auth.ini <<INI
[mlflow]
default_permission = NO_PERMISSIONS
database_uri = sqlite:////mlauth/basic_auth.db
admin_username = admin
admin_password = ${MLFLOW_ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
INI
export MLFLOW_AUTH_CONFIG_PATH=/mlauth/basic_auth.ini
exec mlflow server \
  --app-name basic-auth \
  --backend-store-uri "postgresql+psycopg2://mlflow:${POSTGRES_PASSWORD}@db:5432/mlflow" \
  --artifacts-destination /mlartifacts --serve-artifacts \
  --host 0.0.0.0 --port 5000 --workers 4 \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ORIGINS}"
