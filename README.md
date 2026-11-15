# T-404-LOKA-hpc-scripts

Slurm jobs for the M1 experiments. Each experiment is a plain Python script
(ported from the notebooks in `T-404-LOKA-code/experiments/M1/`) plus a `.sbatch` file to submit it.

```
config.sh            # input/output locations, sourced by every job
setup_env.sh         # optional: install requirements into an existing .venv
experiments/
  order_positions_25p.py   + .sbatch   # herding / k-center on positions, 25%  -> data/ordered-data/*.txt
  order_embeddings_10p.py  + .sbatch   # same methods x 3 embeddings, 10%      -> data/ordered-data/embeddings/*.txt
  train_mse_all.py         + .sbatch   # CNN on 1%..10% prefixes, MSE on E     -> data/ordered-data/results/mse_all/*.csv
  train_mse_forgetting.py  + .sbatch   # same, o.txt ordered by forgetting count -> data/ordered-data/results/mse_all/forgetting.csv
  train_mse_herding_forgetting.py + .sbatch # herding epochs, then forgetting epochs -> data/forgetting_counts/herding_and_forgetting/*.csv
  aggregate.py             + .sbatch   # comparison plots                      -> data/plots/*.png
  forgetting_counts.py     + .sbatch   # forgetting events per state of E      -> data/forgetting_counts/forgetting_e.txt
  partial_forgettable.py   + .sbatch   # CNN on herding subsets w/ x% forgettable -> data/partial_forgettable/<source>/ (results/*.csv, forgetting_counts.txt, herding_order.txt; the last two are reused if present)
  batch_nov15.py           + .sbatch   # experiment_plan.xlsx: ECH/ECK x 5 embeddings x CNN/Residual, +.AE bottleneck variants; records train_mse and mse (E) -> data/batch_nov15/<source>/ (results/<id>.csv, orders/, codes/)
  ae_diagnostics.py        + .sbatch   # why are the .AE rows weak? kNN on codes, dead dims, standardized codes, order-on-codes ablation; reads batch_nov15 codes/orders -> data/batch_nov15/<source>/diagnostics/diagnostics.csv
  common.py                            # shared helpers: mse(), LOKA_FORCE, embeddings, cached orders, autoencoder, 1D models
  cal_herding_kcenter.py   + .sbatch   # CAL active learning inside herding / k-center prefixes -> data/cal_herding_kcenter/<source>/ (results/*.csv, *_order.txt; orders reused if present)
```

## Setup (once, on the login node)

Layout on the cluster (`PROJECT_ROOT` in `config.sh`):
```
/hpchome/axels22/.venv                  # venv with loka + torch installed (see below)
/hpchome/axels22/requirements.txt       # copy of T-404-LOKA-code/requirements.txt, read by setup_env.sh
/hpchome/axels22/data/game_states/o.txt # copy the data over (scp/rsync); e.txt next to it for LOKA_SOURCE=e
/hpchome/axels22/logs/                  # Slurm logs (absolute path in every .sbatch)
```
Results, orders and reports are written under `/hpchome/axels22/data/` and `/hpchome/axels22/reports/`.

Jobs run on `gpu-1xA100` (2-day limit); `gpu-8xA100` allows up to 4 days.
If `PROJECT_ROOT` changes, update the `--output=` path in the `.sbatch` files too (`#SBATCH` lines can't read variables).

### The `loka` package on the cluster

The scripts import `loka` from `T-404-LOKA-code`, which `requirements.txt` pins to a git commit
(`-e git+ssh://...T-404-LOKA-code.git@<commit>#egg=loka`). The cluster only sees code at that commit, so
**whenever the scripts need newer `loka` code, bump the pin and reinstall**:
```
# on your machine: push T-404-LOKA-code, then in its requirements.txt set the new commit, copy it to the cluster
scp requirements.txt <cluster>:/hpchome/axels22/requirements.txt
# on the login node
bash setup_env.sh        # uv pip install -r requirements.txt into .venv
```
`batch_nov15` needs `loka` from branch `m2`, commit `6881ca1` or later (`BreakthroughResNet`, `AutoEncoder`,
`N_BOARD_FEATURES`, `with_clear_lane`). The older pin `b814777` does not have them, and the job fails on import.

Before a long run, check the install inside the venv:
`python -c "from loka.models import BreakthroughResNet, AutoEncoder; from loka.embeddings import with_clear_lane"`.
`check_env.sbatch` only exercises the older experiments, not `batch_nov15`. For that, run a quick pipeline check
(2 epochs, 2 splits, writes to `data/batch_nov15_smoke/`, so it never blocks the real results):
`LOKA_SMOKE=1 LOKA_EXPERIMENTS="ECH1,ECK4.AE" sbatch experiments/batch_nov15.sbatch`

## Running

From the dir that holds `config.sh` and `experiments/`:
```
mkdir -p /hpchome/axels22/logs
sbatch experiments/order_positions_25p.sbatch
sbatch experiments/order_embeddings_10p.sbatch
sbatch experiments/train_mse_all.sbatch     # after both order jobs
sbatch experiments/aggregate.sbatch         # after training
sbatch experiments/forgetting_counts.sbatch # independent; needs data/game_states/e.txt
sbatch experiments/train_mse_herding_forgetting.sbatch   # needs herding_25p.txt and forgetting_o.txt
sbatch experiments/partial_forgettable.sbatch # computes forgetting counts + herding order, or reuses the saved ones
sbatch experiments/cal_herding_kcenter.sbatch # computes herding / k-center orders (or reuses saved), then CAL on each
```

`partial_forgettable` and `cal_herding_kcenter` default to `o.txt`; run on `e.txt` with
`sbatch --export=ALL,LOKA_SOURCE=e experiments/<job>.sbatch` (outputs go to a per-source folder).

`batch_nov15` runs all 40 experiments (ECH1-10, ECK1-10, each also `.AE`). Orders and autoencoder codes are computed once and cached;
`LOKA_FORCE=1` recomputes them and the results. Pick experiments with `LOKA_EXPERIMENTS` (comma separated, wildcards):
`LOKA_EXPERIMENTS="ECH*,ECK3.AE" LOKA_FORCE=1 sbatch experiments/batch_nov15.sbatch`
(set variables in front of `sbatch`; `--export` splits values on commas)

Or chain them so each step waits for the previous one:
```
a=$(sbatch --parsable experiments/order_positions_25p.sbatch)
b=$(sbatch --parsable experiments/order_embeddings_10p.sbatch)
c=$(sbatch --parsable --dependency=afterok:$a:$b experiments/train_mse_all.sbatch)
sbatch --dependency=afterok:$c experiments/aggregate.sbatch
```

Monitor with `squeue -u $USER` and `tail -f /hpchome/axels22/logs/<job>_<id>.out`.

## Notes

- **Constants**: every script starts with a `CONFIG` block. Input/output paths derive from `LOKA_ROOT`, which `config.sh` exports. Experiment settings (ratio, methods, epochs...) are set in the script itself; comment out entries in the method lists to choose what runs.
- **Resumable**: an output that already exists is skipped. To redo a run, delete its `.txt` or `.csv`. If a job times out, resubmit it and it continues with the outputs that are still missing.
- **Local runs**: `LOKA_ROOT=/path/to/codebase python experiments/aggregate.py`.
