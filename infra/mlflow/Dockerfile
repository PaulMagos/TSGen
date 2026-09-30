FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.10 "mlflow[auth]==3.16.1"
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
