#!/bin/bash
# One-time setup on the login node: creates .venv and installs loka + requirements.
# Usage: bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
source config.sh

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$HOME/.uv-python"
export UV_INDEX_STRATEGY=unsafe-best-match

command -v uv >/dev/null || pip install --user uv

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    uv python install 3.12
    uv venv --python 3.12 "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

uv pip install -r "$PROJECT_ROOT/requirements.txt"

mkdir -p "$LOG_DIR" \
    "$LOKA_ROOT/data/ordered-data/embeddings" \
    "$LOKA_ROOT/data/ordered-data/results/mse_all" \
    "$LOKA_ROOT/data/plots"

python -c "import loka, torch; print('loka ok, torch', torch.__version__)"
