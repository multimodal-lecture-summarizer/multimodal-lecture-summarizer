#!/bin/sh
# Blocks until the bundled services the backend and the worker depend on accept connections.
# Starting before Chroma is up would make the process settle on the on-disk fallback for good.
until /usr/lib/postgresql/15/bin/pg_isready -h 127.0.0.1 -q; do sleep 1; done

if [ "$AIO_START_CHROMA" = "true" ]; then
    until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/v2/heartbeat', timeout=2)" > /dev/null 2>&1; do
        sleep 1
    done
fi
