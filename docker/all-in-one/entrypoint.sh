#!/bin/sh
# Prepares the data volumes and the environment, then hands over to supervisord.
#   - First start: initialises PostgreSQL on the data volume and creates the application database.
#   - Points the backend and worker at that database (overriding any hosted database in backend/.env).
#   - Starts the bundled Redis, unless SHARED_REDIS_URL is set (e.g. so a Colab worker can join).
set -e

PGBIN=/usr/lib/postgresql/15/bin
PGDATA=/var/lib/postgresql/data
PGSOCKET=/var/run/postgresql
DB_NAME="${AIO_POSTGRES_DB:-video_summarizer}"
DB_USER="${AIO_POSTGRES_USER:-postgres}"
DB_PASS="${AIO_POSTGRES_PASSWORD:-postgres}"

as_postgres() { runuser -u postgres -- "$@"; }

mkdir -p "$PGDATA" "$PGSOCKET" /var/lib/redis /app/storage /app/cache /app/outputs
chown -R postgres:postgres "$PGDATA" "$PGSOCKET"
chown -R redis:redis /var/lib/redis
chmod 700 "$PGDATA"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    echo "[all-in-one] Initialising PostgreSQL in $PGDATA ..."
    pwfile=$(mktemp)
    printf '%s' "$DB_PASS" > "$pwfile"
    chown postgres "$pwfile"
    as_postgres "$PGBIN/initdb" -D "$PGDATA" --username="$DB_USER" --pwfile="$pwfile" \
        --auth-local=trust --auth-host=scram-sha-256 --encoding=UTF8 > /dev/null
    rm -f "$pwfile"

    # Start a private instance (socket only) just to create the application database.
    as_postgres "$PGBIN/pg_ctl" -D "$PGDATA" -w \
        -o "-c listen_addresses='' -c unix_socket_directories=$PGSOCKET" start > /dev/null
    if ! as_postgres "$PGBIN/psql" -h "$PGSOCKET" -U "$DB_USER" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
        as_postgres "$PGBIN/createdb" -h "$PGSOCKET" -U "$DB_USER" "$DB_NAME"
    fi
    as_postgres "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop > /dev/null
    echo "[all-in-one] Database '$DB_NAME' is ready."
fi

# The bundled database wins over any hosted POSTGRES_* values that came in with backend/.env.
export POSTGRES_SERVER=localhost POSTGRES_PORT=5432
export POSTGRES_USER="$DB_USER" POSTGRES_PASSWORD="$DB_PASS" POSTGRES_DB="$DB_NAME"
unset POSTGRES_SSLMODE
export DATABASE_URL="postgresql://$DB_USER:$DB_PASS@127.0.0.1:5432/$DB_NAME"

if [ -n "$SHARED_REDIS_URL" ]; then
    export CELERY_BROKER_URL="$SHARED_REDIS_URL" CELERY_RESULT_BACKEND="$SHARED_REDIS_URL" REDIS_URL="$SHARED_REDIS_URL"
    export AIO_START_REDIS=false
    echo "[all-in-one] Using SHARED_REDIS_URL; the bundled Redis is not started."
else
    export CELERY_BROKER_URL=redis://127.0.0.1:6379/0 CELERY_RESULT_BACKEND=redis://127.0.0.1:6379/1
    export REDIS_URL=redis://127.0.0.1:6379/0
    export AIO_START_REDIS=true
fi
# Chroma runs here as a server (one process owns the vector data; the API and the worker are
# both HTTP clients), unless AIO_EXTERNAL_CHROMA=true keeps the CHROMADB_* values of backend/.env.
if [ "$AIO_EXTERNAL_CHROMA" = "true" ]; then
    export AIO_START_CHROMA=false
    echo "[all-in-one] AIO_EXTERNAL_CHROMA=true; using the Chroma server from backend/.env."
else
    mkdir -p /var/lib/chroma
    export CHROMADB_HOST=127.0.0.1 CHROMADB_PORT=8001 CHROMADB_SSL=False CHROMADB_API_KEY=""
    export CHROMADB_TENANT=default_tenant CHROMADB_DATABASE=default_database
    export AIO_START_CHROMA=true
fi

export WORKER_QUEUES="${WORKER_QUEUES:-celery}"

exec "$@"
