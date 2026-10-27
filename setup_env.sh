#!/bin/bash
# One-time setup on the login node: installs loka + requirements into the existing .venv.
# Usage: bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"
source config.sh

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "no venv at $VENV_DIR (expected it to exist already)" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

pip install -r "$PROJECT_ROOT/requirements.txt"

mkdir -p "$LOG_DIR" \
         "$LOKA_ROOT/data/ordered-data/embeddings" \
         "$LOKA_ROOT/data/ordered-data/results/mse_all" \
         "$LOKA_ROOT/data/plots"

python -c "import loka, torch; print('loka ok, cuda available:', torch.cuda.is_available())"
