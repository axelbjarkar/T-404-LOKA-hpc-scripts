# Shared input/output locations for every job. Edit these once on the cluster.
# Sourced by setup_env.sh and every experiments/*.sbatch.

export PROJECT_ROOT="/hpchome/axels22"

# This hpc-scripts dir, wherever it's cloned (Slurm logs go in here)
export LOG_DIR="$PROJECT_ROOT/logs"

# Python environment
export VENV_DIR="$PROJECT_ROOT/.venv"

# Read by the experiment scripts: every input/output path is derived from this
export LOKA_ROOT="$PROJECT_ROOT"
