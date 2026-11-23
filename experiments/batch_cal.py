"""CAL batch (sheet 'Loka experiments.xlsx', "CAL from penultimate layer"): o.txt (5x5), position-only features. Every subset starts as a
prefix of the cached herding / k-center order (the same orders as the ECH / ECK rows of batch_nov15) and the rest of the budget is
acquired with CAL from the WHOLE pool (E minus the labeled states), not from the order's prefix as in the old CAL runs.

  CAL{C|R}{H|K}{01-04}   model C = CNN, R = Residual (parameter matched to the CNN)   start method H = herding, K = k-center
                         starting size 35, 50, 65, 80 % of each split's budget  (01 .. 04)
  CALC{H|K}{01-04}R      control (CNN only): same start, but the acquired states are drawn uniformly from the pool

For every split (1%..10% of E, 10 splits) with n = split * |E| labels:
  1. labeled = the first round(start * n) states of the order
  2. ROUNDS times: train a fresh model (50 epochs, same recipe as the other rows) on `labeled` only to score the pool, give every pool
     state its CAL score (mean |prediction of its K nearest labeled states - its own prediction|, distances on the input of the model's
     last Linear layer = penultimate layer, labels of the pool are never used), and label the q highest scores
  3. after the last round labeled has exactly n states; train a fresh model (50 epochs) on it and record the MSE on it (`train_mse`)
     and on all of E (`mse`) -> results/<id>.csv with columns ratio, mse, train_mse, acquired_hard_share
Fairness: every row spends n labels and its final model is trained exactly like the ECH / ECK rows (fresh, 50 epochs, same settings);
the intermediate scoring models are only used to choose states and are not carried over. The scoring model has the architecture of the
final one. `acquired_hard_share` = share of the acquired states whose forgetting count is >= 11 (only if the counts file exists, else NaN).
Settings by environment: LOKA_EXPERIMENTS="CALCH0?,CALRK04" (comma separated, wildcards), LOKA_CAL_ROUNDS (default 5), LOKA_CAL_K (20),
LOKA_CAL_CHUNK (pool states per distance block, default 2048, lower it on a small GPU), LOKA_SEED (default 42, others write
<id>.s<seed>.csv), LOKA_ORDERS_DIR (default $LOKA_ROOT/data/batch_nov15/<source>/orders), LOKA_FORCE=1, LOKA_SMOKE=1
(first 100k states, 2 epochs, 2 rounds, own directory and orders)."""
import fnmatch
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from common import board_size_of, build_model, fresh, get_order, mse
from loka.data import load_breakthrough_states
from loka.report import ExperimentReport, config_from_globals
from loka.training_methods import CAL_scores

# ---- CONFIG: input / output locations ----
ROOT           = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE         = os.environ.get("LOKA_SOURCE", "o")
SOURCE_STATES  = ROOT / "data" / "game_states" / f"{SOURCE}.txt"
FORGETTING     = ROOT / "data" / "forgetting_counts" / f"forgetting_{SOURCE}.txt"   # optional, only for acquired_hard_share
SMOKE          = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")   # tiny run to check the pipeline; own directory
EXPERIMENT_DIR = ROOT / ("data/batch_cal_smoke" if SMOKE else "data/batch_cal") / SOURCE
RESULTS_DIR    = EXPERIMENT_DIR / "results"     # <id>.csv
ORDERS_DIR     = Path(os.environ.get("LOKA_ORDERS_DIR", EXPERIMENT_DIR / "orders" if SMOKE else ROOT / "data/batch_nov15" / SOURCE / "orders"))
N_SMOKE        = 100_000

# ---- CONFIG: experiment settings (as in batch_nov15) ----
START = 0.01
END = 0.1
STEPS = 10
EPOCHS = 50
BATCH_SIZE = 512
EVAL_BS = 8192
LR = 0.001
SEED = int(os.environ.get("LOKA_SEED", "42"))
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name
ROUNDS = int(os.environ.get("LOKA_CAL_ROUNDS", "5"))
K_NEAREST = int(os.environ.get("LOKA_CAL_K", "20"))
CAL_CHUNK = int(os.environ.get("LOKA_CAL_CHUNK", "2048"))
SPACE = "penultimate"
HARD = 11            # forgetting count that counts as hard
if SMOKE:
    EPOCHS, STEPS, ROUNDS = 2, 2, 2

STARTS = [0.35, 0.50, 0.65, 0.80]                       # experiment 01 .. 04
EXPERIMENTS = {}                                          # id -> setup
for m_code, model in (('C', 'cnn'), ('R', 'residual')):
    for k_code, method in (('H', 'herding'), ('K', 'k_center')):
        for i, start in enumerate(STARTS, start=1):
            EXPERIMENTS[f'CAL{m_code}{k_code}{i:02d}'] = dict(method=method, model=model, start=start, acquire='cal')
            if m_code == 'C':                              # random-pool controls, CNN only
                EXPERIMENTS[f'CAL{m_code}{k_code}{i:02d}R'] = dict(method=method, model=model, start=start, acquire='random')

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def selected():
    patterns = [p.strip() for p in os.environ.get('LOKA_EXPERIMENTS', '*').split(',') if p.strip()]
    names = [n for n in EXPERIMENTS if any(fnmatch.fnmatchcase(n, p) for p in patterns)]
    if not names:
        raise SystemExit(f'LOKA_EXPERIMENTS={patterns} matches nothing; known: {list(EXPERIMENTS)}')
    return names


def csv_path(name):
    return RESULTS_DIR / (f'{name}.csv' if SEED == 42 else f'{name}.s{SEED}.csv')


def read_counts(path, n):
    with open(path) as f:
        return np.fromiter((int(line.rsplit(None, 1)[-1]) for line in f), dtype=np.int64, count=n)


def train(model, optimizer, loss_fn, X, Y):
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), device=device)   # new shuffle each epoch
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            optimizer.step()


def fit(kind, X, Y, board_size):
    """A fresh model trained like every other row of the experiments."""
    torch.manual_seed(SEED)
    model = build_model(kind, X, on_code=False, board_size=board_size).to(device)
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    train(model, optimizer, getattr(nn, LOSS)(), X, Y)
    return model, optimizer


def acquire_cal(kind, X_t, Y_t, labeled, n, board_size):
    """Grow `labeled` (indexes into E) to n states: ROUNDS x (fit on labeled, score the pool, add the highest scores)."""
    quota = np.diff(np.linspace(0, n - len(labeled), ROUNDS + 1).round().astype(int))
    for q in quota:
        if q == 0:
            continue
        model, optimizer = fit(kind, X_t[torch.as_tensor(labeled, device=device)], Y_t[torch.as_tensor(labeled, device=device)], board_size)
        in_labeled = np.zeros(len(X_t), dtype=bool)
        in_labeled[labeled] = True
        pool = np.flatnonzero(~in_labeled)
        scores = CAL_scores(model, X_t[torch.as_tensor(labeled, device=device)], X_t[torch.as_tensor(pool, device=device)],
                            K_NEAREST, chunk=CAL_CHUNK, bs=EVAL_BS, space=SPACE)
        top = scores.topk(int(q)).indices.cpu().numpy()
        labeled = np.concatenate([labeled, pool[top]])
        del model, optimizer, scores
        torch.cuda.empty_cache()
    return labeled


def run_one(rep, name, exp, X_flat, X_t, Y_t, board_size, counts, random_order):
    method, kind, start = exp['method'], exp['model'], exp['start']
    order = get_order(ORDERS_DIR / f'{method}_position_only.txt', X_flat, method, END, device, rep)
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    rows = []

    pbar = tqdm(total=len(splits), desc=name)
    for m, s in enumerate(splits):
        n = int(s * len(X_t))
        n0 = int(round(start * n))
        base = np.asarray(order[:n0])
        if exp['acquire'] == 'cal':
            labeled = acquire_cal(kind, X_t, Y_t, base, n, board_size)
        else:   # random states of the pool (not in the prefix), nested over the splits
            taken = np.zeros(len(X_t), dtype=bool)
            taken[base] = True
            labeled = np.concatenate([base, random_order[~taken[random_order]][:n - n0]])
        assert len(labeled) == n and len(np.unique(labeled)) == n

        idx = torch.as_tensor(labeled, device=device)
        X_S, Y_S = X_t[idx], Y_t[idx]
        model, optimizer = fit(kind, X_S, Y_S, board_size)
        if m == 0:
            info = rep.add_model(name, model, input_shape=list(X_S.shape[1:]), embedding='position_only', method=method,
                                 board_size=board_size, start=start, acquire=exp['acquire'])
            print(f'{name}: {kind}, {info["n_params"]:,} parameters, start {start:.0%}, acquire {exp["acquire"]}, {ROUNDS} rounds', flush=True)
        train_mse = mse(model, X_S, Y_S, EVAL_BS)   # on the states it was trained on
        e_mse = mse(model, X_t, Y_t, EVAL_BS)       # on all of E
        acquired = labeled[n0:]
        hard = float((counts[acquired] >= HARD).mean()) if counts is not None and len(acquired) else float('nan')
        rows.append({'ratio': s, 'mse': e_mse, 'train_mse': train_mse, 'acquired_hard_share': hard})
        rep.log_metric(f'{name}/mse_E', e_mse, step=float(s))
        rep.log_metric(f'{name}/mse_train', train_mse, step=float(s))

        del model, optimizer, X_S, Y_S
        torch.cuda.empty_cache()
        pbar.update(1)
        pbar.set_postfix(split=f'{s:.2f}', train=f'{train_mse:.5f}', E=f'{e_mse:.5f}', hard=f'{hard:.2f}')
    pbar.close()

    pd.DataFrame(rows).to_csv(csv_path(name), index=False)
    print(f'wrote {csv_path(name)}', flush=True)
    rep.add_output(csv_path(name), input='source_states (E)', **exp)


def main():
    print(device, f'| source {SOURCE_STATES} | seed {SEED} | {ROUNDS} rounds, k={K_NEAREST}, {SPACE}', '| FORCE' if os.environ.get('LOKA_FORCE', '0') not in ('', '0') else '',
          '| SMOKE' if SMOKE else '', flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('batch_cal', config=config_from_globals(globals())) as rep:
        todo = []
        for name in selected():
            if fresh(csv_path(name)):
                todo.append(name)
            else:
                print(f'skipping {name}: {csv_path(name)} already exists')
                rep.add_skipped(csv_path(name))
        if not todo:
            return

        rep.add_data('source_states (E)', SOURCE_STATES)
        X_flat, Y = load_breakthrough_states(SOURCE_STATES)   # (N, 2 * side^2) positions
        if SMOKE:
            X_flat, Y = X_flat[:N_SMOKE], Y[:N_SMOKE]
        board_size = board_size_of(X_flat)
        counts = None
        if FORGETTING.exists():
            counts = read_counts(FORGETTING, len(X_flat))
            rep.add_data('forgetting counts', FORGETTING)
        else:
            print(f'{FORGETTING} not found: acquired_hard_share will be NaN', flush=True)
        print(f'loaded {len(X_flat)} states', flush=True)

        random_order = np.random.default_rng(1).permutation(len(X_flat))   # random control: fixed, independent of the training seed
        X_t = torch.as_tensor(X_flat, dtype=torch.float32, device=device)   # features the models train on
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)

        for name in todo:
            run_one(rep, name, EXPERIMENTS[name], X_flat, X_t, Y_t, board_size, counts, random_order)


if __name__ == '__main__':
    main()
