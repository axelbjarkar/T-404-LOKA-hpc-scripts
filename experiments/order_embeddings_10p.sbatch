#!/bin/bash
#SBATCH --job-name=order_embeddings_10p
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Submit from the hpc-scripts dir: sbatch experiments/order_embeddings_10p.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=60  # one progress line per minute instead of flooding the log

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT"
nvidia-smi

python experiments/order_embeddings_10p.py
