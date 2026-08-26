#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    bash setup_macos_linux.sh
fi

exec ".venv/bin/python" app.py
