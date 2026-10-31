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
  aggregate.py             + .sbatch   # comparison plots                      -> data/plots/*.png
  forgetting_counts.py     + .sbatch   # forgetting events per state of E      -> data/forgetting_counts/forgetting_e.txt
```

## Setup (once, on the login node)

Layout on the cluster (`PROJECT_ROOT` in `config.sh`):
```
/hpchome/axels22/.venv        # existing venv with loka + torch installed
/hpchome/axels22/data/o.txt   # copy the data over (scp/rsync)
/hpchome/axels22/logs/        # Slurm logs (absolute path in every .sbatch)
```

Jobs run on `gpu-1xA100` (2-day limit); `gpu-8xA100` allows up to 4 days.
If `PROJECT_ROOT` changes, update the `--output=` path in the `.sbatch` files too (`#SBATCH` lines can't read variables).

## Running

From the dir that holds `config.sh` and `experiments/`:
```
mkdir -p /hpchome/axels22/logs
sbatch experiments/order_positions_25p.sbatch
sbatch experiments/order_embeddings_10p.sbatch
sbatch experiments/train_mse_all.sbatch     # after both order jobs
sbatch experiments/aggregate.sbatch         # after training
sbatch experiments/forgetting_counts.sbatch # independent; needs data/game_states/e.txt
```

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
