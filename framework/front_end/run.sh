#!/bin/bash

source /app/.venv/bin/activate

if [ -z "$1" ]; then
    exec streamlit run ./0_Home.py --server.port 8001
else
  echo "$@"
  exec "$@"
fi



