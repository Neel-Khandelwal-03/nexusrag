#!/bin/sh
# Prepare the writable paths, index the sample documents on first start, then run the
# command (Chainlit by default). Indexing failures are logged, not fatal: the app starts
# and explains the problem in the chat.
set -eu

mkdir -p "${STORAGE_DIR:-/data/storage}" "${HF_HOME:-/data/models}"

if [ "${BOOTSTRAP_INDEX:-false}" = "true" ]; then
    python /app/scripts/bootstrap_index.py || true
fi

# Chainlit takes the port as an argument, so append it unless the caller set one.
case " $* " in
    *" --port "*) exec "$@" ;;
    *) exec "$@" --port "${PORT:-8000}" ;;
esac
