#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  .venv/bin/python app.py
else
  echo "Local .venv was not found. Running setup_mac.sh first..."
  ./setup_mac.sh
  .venv/bin/python app.py
fi
