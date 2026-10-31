#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --partition=any_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/hpchome/axels22/logs/%x_%j.out   # must match LOG_DIR in config.sh; the dir must exist
# Quick check before the real jobs: environment + a tiny run of all four experiments.
# Submit from the hpc-scripts dir: sbatch experiments/check_env.sbatch
# Look for "CHECK OK" at the end of the log.
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
source config.sh
source "$VENV_DIR/bin/activate"

echo "== environment"
echo "host:        $(hostname)"
echo "VIRTUAL_ENV: ${VIRTUAL_ENV:-<unset>}"
echo "PATH:        $PATH"
ls -l "$VENV_DIR"/bin/python*
cat "$VENV_DIR/pyvenv.cfg"
module list 2>&1 || true
echo "python  -> $(command -v python || echo NOT FOUND)"
echo "python3 -> $(command -v python3 || echo NOT FOUND)"

set -e  # from here on, any failure stops the check

echo "== imports"
python -c "import loka, torch, torchvision, numpy, pandas, matplotlib, tqdm; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

echo "== data"
ls -l "$LOKA_ROOT/data/game_states/o.txt"

echo "== tiny end-to-end run"
TMP="${TMPDIR:-/tmp}/check_env_$SLURM_JOB_ID"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/data/game_states" "$TMP/exp"
head -5000 "$LOKA_ROOT/data/game_states/o.txt" > "$TMP/data/game_states/o.txt"
cp experiments/*.py "$TMP/exp/"
sed -i 's/^STEPS = .*/STEPS = 2/; s/^EPOCHS = .*/EPOCHS = 2/' "$TMP/exp/train_mse_all.py"

for script in order_positions_25p order_embeddings_10p train_mse_all aggregate; do
    echo "-- $script"
    LOKA_ROOT="$TMP" TQDM_DISABLE=1 python "$TMP/exp/$script.py" | tail -3
done

echo "outputs: $(find "$TMP/data" -type f | wc -l) files (expect 28)"

echo "== reports"
n_reports=$(find "$TMP/reports" -name report.json | wc -l)
echo "reports: $n_reports (expect 4), index entries: $(wc -l < "$TMP/reports/index.jsonl") (expect 27)"
bad=$(grep -L '"status": "ok"' "$TMP"/reports/*/report.json || true)
test -z "$bad" || { echo "reports not ok: $bad"; exit 1; }
test "$n_reports" -eq 4
echo "CHECK OK"
