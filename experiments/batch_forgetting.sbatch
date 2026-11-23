#!/bin/bash
#SBATCH --job-name=batch_forgetting
#SBATCH --partition=gpu-1xA100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Forgetting batch FGCH/FGCK/FGRH/FGRK 01-06 (+ random controls) on o.txt: p% of each subset = highest-forgetting-count states. Needs data/forgetting_counts/forgetting_o.txt and the cached batch_nov15 orders.
# Submit from the hpc-scripts dir: sbatch experiments/batch_forgetting.sbatch
#   a subset:         LOKA_EXPERIMENTS="FGCH02,FGRK06" sbatch experiments/batch_forgetting.sbatch
#   recompute everything (results, the hardest-first order AND the shared batch_nov15 herding / k-center orders): LOKA_FORCE=1 sbatch experiments/batch_forgetting.sbatch
#   quick pipeline check (2 epochs, own data dir): LOKA_SMOKE=1 LOKA_EXPERIMENTS="FGCH02,FGRK06" sbatch experiments/batch_forgetting.sbatch
# (sbatch passes the environment on; do not use --export for these, it splits values on commas)
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

export TQDM_MININTERVAL=10  # progress update at most every 10 s

echo "job $SLURM_JOB_ID on $(hostname), LOKA_ROOT=$LOKA_ROOT, LOKA_EXPERIMENTS=${LOKA_EXPERIMENTS:-*}, LOKA_FORCE=${LOKA_FORCE:-0}"
nvidia-smi

python experiments/batch_forgetting.py
