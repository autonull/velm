#!/usr/bin/env bash
set -euo pipefail

# tuner.sh — launch the Optuna + Rich tuner from the repo without installing the package.
# Usage: ./tuner.sh --trials 20 --steps 100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ensure local src/ is on PYTHONPATH so 'import velm' works
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

PY="${PYTHON:-python}"

# Quick dependency check (optuna, rich)
if ! ${PY} -c "import optuna, rich" >/dev/null 2>&1; then
  echo "One or more tuner dependencies (optuna, rich) are missing."
  echo "Install dev dependencies: pip install -e .[dev] or pip install optuna rich"
fi

# Run the tuner script directly (no -m, no package import tricks)
exec ${PY} "${SCRIPT_DIR}/src/tools/tuner.py" "$@"
