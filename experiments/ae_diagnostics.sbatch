#!/bin/bash
#SBATCH --job-name=ae_diagnostics
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Diagnostics for the .AE rows; needs the codes/ and orders/ written by batch_nov15 (trains no autoencoder).
# Submit from the hpc-scripts dir: sbatch experiments/ae_diagnostics.sbatch
#   some embeddings: LOKA_EMBEDDINGS="near_goal,clear_lane" sbatch experiments/ae_diagnostics.sbatch
#   other ratio:     LOKA_RATIO=0.02 sbatch experiments/ae_diagnostics.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EMBEDDINGS=${LOKA_EMBEDDINGS:-all}, LOKA_RATIO=${LOKA_RATIO:-0.05}"
nvidia-smi

python experiments/ae_diagnostics.py
