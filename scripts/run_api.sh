#!/usr/bin/env bash
# Launch the FastAPI wrapper on port 3001 (matches NEXT_PUBLIC_API_URL).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/worker/src:${PYTHONPATH:-}"
exec uvicorn api.main:app --host 0.0.0.0 --port 3001 --reload
