#!/bin/bash
#SBATCH --job-name=batch_nov15
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Submit from the hpc-scripts dir: sbatch experiments/batch_nov15.sbatch
#   a subset:         LOKA_EXPERIMENTS="ECH*,ECK1.AE" sbatch experiments/batch_nov15.sbatch
#   recompute orders: LOKA_FORCE=1 sbatch experiments/batch_nov15.sbatch
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_EXPERIMENTS="ECH1,ECK4.AE" sbatch experiments/batch_nov15.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EXPERIMENTS=${LOKA_EXPERIMENTS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/batch_nov15.py
