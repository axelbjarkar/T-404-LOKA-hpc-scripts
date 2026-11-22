#!/bin/bash
#SBATCH --job-name=board_comparison
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# 6x6 board comparison BC1-BC4. Needs /hpchome/axels22/data/game_states/bt6x6_fens.txt and loka with board_size support.
# Submit from the hpc-scripts dir: sbatch experiments/board_comparison.sbatch
#   a subset:         LOKA_EXPERIMENTS="BC1,BC4" sbatch experiments/board_comparison.sbatch
#   recompute orders: LOKA_FORCE=1 sbatch experiments/board_comparison.sbatch
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_EXPERIMENTS="BC1,BC4" sbatch experiments/board_comparison.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EXPERIMENTS=${LOKA_EXPERIMENTS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/board_comparison.py
