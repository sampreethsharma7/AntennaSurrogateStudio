#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

echo
echo "============================================================"
echo "  Antenna Surrogate Studio - first-time setup"
echo "============================================================"
echo

PYTHON_CMD=""
for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys, struct; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) and struct.calcsize("P") == 8 else 1)' >/dev/null 2>&1; then
        PYTHON_CMD="$candidate"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "64-bit Python 3.11, 3.12, or 3.13 is required."
    echo "Install Python from https://www.python.org/downloads/ and run this again."
    exit 1
fi

echo "Using $("$PYTHON_CMD" --version)"

if ! "$PYTHON_CMD" -c "import tkinter" >/dev/null 2>&1; then
    echo
    echo "Python's Tk desktop support is missing."
    echo "On Debian/Ubuntu, install python3-tk. On macOS, use Python from python.org."
    exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating the private application environment..."
    "$PYTHON_CMD" -m venv .venv
fi

echo "Installing the streamlined desktop requirements..."
".venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
".venv/bin/python" -m pip install --disable-pip-version-check -r requirements.txt
".venv/bin/python" -c "import tkinter, customtkinter, numpy, sklearn, scipy, joblib, xgboost"

echo
echo "Setup complete."
echo "Run: bash start_studio.sh"
