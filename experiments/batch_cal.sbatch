#!/bin/bash
#SBATCH --job-name=batch_cal
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# CAL batch CALCH/CALCK/CALRH/CALRK 01-04 (+ CALCH/CALCK 01-04 R random-pool controls) on o.txt: herding / k-center start, rest acquired with CAL (penultimate layer) from the whole pool. Needs loka >= 2afdd5e (CAL_scores) and the cached batch_nov15 orders.
# Submit from the hpc-scripts dir: sbatch experiments/batch_cal.sbatch
#   a subset:         LOKA_EXPERIMENTS="CALCH01,CALRK04" sbatch experiments/batch_cal.sbatch
#   rounds / chunk:   LOKA_CAL_ROUNDS=5 LOKA_CAL_CHUNK=2048 sbatch experiments/batch_cal.sbatch   (more rounds = more adaptive, but each round trains a model)
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_EXPERIMENTS="CALCH01,CALRK04" sbatch experiments/batch_cal.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EXPERIMENTS=${LOKA_EXPERIMENTS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/batch_cal.py
