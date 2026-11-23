#!/bin/bash
#SBATCH --job-name=error_analysis
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Error on E by forgetting count (CNN and Residual, herding 10%); needs data/forgetting_counts/forgetting_o.txt and the cached batch_nov15 orders.
# Submit from the hpc-scripts dir: sbatch experiments/error_analysis.sbatch
#   a subset:         LOKA_MODELS="cnn" sbatch experiments/error_analysis.sbatch
#   + CAL check (does a label-free CAL score find the hard states?): LOKA_CAL=1 sbatch experiments/error_analysis.sbatch   (LOKA_CAL_SPACES="raw,penultimate", LOKA_CAL_K=20)
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_MODELS="cnn" sbatch experiments/error_analysis.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_MODELS=${LOKA_MODELS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/error_analysis.py
