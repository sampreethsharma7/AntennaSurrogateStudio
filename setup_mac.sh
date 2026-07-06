#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "Antenna Surrogate Studio - macOS Setup"
echo "============================================================"
echo
echo "This setup will:"
echo "  1. Find Python 3.10 or newer"
echo "  2. Create a local .venv environment"
echo "  3. Install required packages from requirements.txt"
echo

find_python() {
  for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
      version="$($cmd - <<'PYVER'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PYVER
)"
      major="${version%%.*}"
      minor="${version#*.}"
      if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_CMD="$(find_python || true)"
if [ -z "$PYTHON_CMD" ]; then
  echo "Python 3.10 or newer was not found."
  echo "Install Python 3.11 or 3.12 from https://www.python.org/downloads/macos/ or Homebrew, then run this file again."
  echo
  read -r -p "Press Enter to exit..."
  exit 1
fi

echo "Python found: $($PYTHON_CMD --version)"
echo

if [ -x ".venv/bin/python" ]; then
  echo "Existing .venv environment found. It will be reused."
else
  "$PYTHON_CMD" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete. Double-click run_app.command to launch Antenna Surrogate Studio."
echo
read -r -p "Press Enter to close this window..."
