#!/bin/bash
#SBATCH --job-name=resnet_size
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# ResNet size comparison RSZH01-04, RSZK01-04 on o.txt (extra residual blocks).
# Submit from the hpc-scripts dir: sbatch experiments/resnet_size.sbatch
#   a subset:         LOKA_EXPERIMENTS="RSZH01,RSZK04" sbatch experiments/resnet_size.sbatch
#   recompute orders: LOKA_FORCE=1 sbatch experiments/resnet_size.sbatch
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_EXPERIMENTS="RSZH01,RSZK04" sbatch experiments/resnet_size.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EXPERIMENTS=${LOKA_EXPERIMENTS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/resnet_size.py
