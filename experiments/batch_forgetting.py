"""Forgetting batch (sheet 'Loka experiments.xlsx', "Forgetting is chosen from max forgotten states"): o.txt (5x5), position-only
features, a share p of every subset is made of the states with the HIGHEST forgetting count, the rest comes from the cached
herding / k-center order (the same orders as the ECH / ECK rows of batch_nov15).

  FG{C|R}{H|K}{01-06}   model C = CNN, R = Residual (parameter matched to the CNN)   method H = herding, K = k-center
                        p = 5, 10, 20, 30, 40, 50 % of the budget  (01 .. 06)
  FGCH{02,04,06}R   control (herding + CNN only): same p, but the swapped-in states are drawn uniformly from E instead of by forgetting count
                        (does the swap itself, or the hardness of the states, change the MSE?)

For every split (1%..10% of E, 10 splits) with n = split * |E| states: the first n - k states of the order plus k = round(p * n) other states
(highest count first, ties broken randomly with a fixed seed, states already in the prefix are skipped). The budget is the same as in
the ECH / ECK rows, which are the p = 0 reference. A fresh model is trained per split; we record the MSE on the training subset
(`train_mse`) and on all of E (`mse`) -> results/<id>.csv with columns ratio, mse, train_mse.

The forgetting counts need every label of E (data/forgetting_counts/forgetting_<source>.txt, one row per state of E, same order,
count = last field): this is an oracle, an upper bound for what hard-state selection can give.
Settings by environment: LOKA_EXPERIMENTS="FGCH0?,FGRK06" (comma separated, wildcards), LOKA_SEED (default 42; other seeds write
<id>.s<seed>.csv), LOKA_ORDERS_DIR (default $LOKA_ROOT/data/batch_nov15/<source>/orders), LOKA_FORCE=1, LOKA_SMOKE=1
(first 100k states, 2 epochs, own directory and orders)."""
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

# ---- CONFIG: input / output locations ----
ROOT           = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE         = os.environ.get("LOKA_SOURCE", "o")
SOURCE_STATES  = ROOT / "data" / "game_states" / f"{SOURCE}.txt"
FORGETTING     = ROOT / "data" / "forgetting_counts" / f"forgetting_{SOURCE}.txt"
SMOKE          = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")   # tiny run to check the pipeline; own directory
EXPERIMENT_DIR = ROOT / ("data/batch_forgetting_smoke" if SMOKE else "data/batch_forgetting") / SOURCE
RESULTS_DIR    = EXPERIMENT_DIR / "results"     # <id>.csv
ORDERS_DIR     = Path(os.environ.get("LOKA_ORDERS_DIR", EXPERIMENT_DIR / "orders" if SMOKE else ROOT / "data/batch_nov15" / SOURCE / "orders"))
HARD_ORDER     = EXPERIMENT_DIR / "orders" / "forgetting_hardest_first.txt"   # cached: indexes into E, highest count first
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
TIE_SEED = 0         # tie-break among equal forgetting counts, and the random control: fixed, independent of the training seed
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name
if SMOKE:
    EPOCHS, STEPS = 2, 2

SHARES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]           # experiment 01 .. 06
EXPERIMENTS = {}                                          # id -> setup
for m_code, model in (('C', 'cnn'), ('R', 'residual')):
    for k_code, method in (('H', 'herding'), ('K', 'k_center')):
        for i, share in enumerate(SHARES, start=1):
            EXPERIMENTS[f'FG{m_code}{k_code}{i:02d}'] = dict(method=method, model=model, share=share, select='forgetting')
            if (m_code, k_code) == ('C', 'H') and i in (2, 4, 6):   # controls only for herding + CNN
                EXPERIMENTS[f'FG{m_code}{k_code}{i:02d}R'] = dict(method=method, model=model, share=share, select='random')

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


def hardest_first(counts):
    """Indexes of E sorted by forgetting count, highest first; states with the same count in a fixed random order."""
    tie = np.random.default_rng(TIE_SEED).permutation(len(counts))
    return tie[np.argsort(-counts[tie], kind='stable')]


def build_subset(order, extra_order, n, share):
    """First n - k states of `order` + the first k states of `extra_order` that are not already in the prefix; k = round(share * n)."""
    k = int(round(share * n))
    base = order[:n - k]
    taken = np.zeros(len(extra_order), dtype=bool)
    taken[base] = True
    extra = extra_order[~taken[extra_order]][:k]
    idx = np.concatenate([base, extra])
    assert len(idx) == n and len(np.unique(idx)) == n
    return idx


def train(model, optimizer, loss_fn, X, Y):
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), device=device)   # new shuffle each epoch
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            optimizer.step()


def run_one(rep, name, exp, X_flat, X_t, Y_t, board_size, extra_orders):
    method, kind, share = exp['method'], exp['model'], exp['share']
    order = get_order(ORDERS_DIR / f'{method}_position_only.txt', X_flat, method, END, device, rep)
    extra_order = extra_orders[exp['select']]
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    rows = []

    pbar = tqdm(total=len(splits), desc=name)
    for m, s in enumerate(splits):
        idx = torch.as_tensor(build_subset(order, extra_order, int(s * len(X_t)), share), device=device)
        X_S, Y_S = X_t[idx], Y_t[idx]

        torch.manual_seed(SEED)
        model = build_model(kind, X_S, on_code=False, board_size=board_size).to(device)
        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            info = rep.add_model(name, model, input_shape=list(X_S.shape[1:]), embedding='position_only',
                                 method=method, board_size=board_size, share=share, select=exp['select'])
            print(f'{name}: {kind}, {info["n_params"]:,} parameters, {share:.0%} {exp["select"]}', flush=True)

        train(model, optimizer, loss_fn, X_S, Y_S)
        train_mse = mse(model, X_S, Y_S, EVAL_BS)   # on the states it was trained on
        e_mse = mse(model, X_t, Y_t, EVAL_BS)       # on all of E
        rows.append({'ratio': s, 'mse': e_mse, 'train_mse': train_mse})
        rep.log_metric(f'{name}/mse_E', e_mse, step=float(s))
        rep.log_metric(f'{name}/mse_train', train_mse, step=float(s))

        del model, optimizer, X_S, Y_S
        torch.cuda.empty_cache()
        pbar.update(1)
        pbar.set_postfix(split=f'{s:.2f}', train=f'{train_mse:.5f}', E=f'{e_mse:.5f}')
    pbar.close()

    pd.DataFrame(rows).to_csv(csv_path(name), index=False)
    print(f'wrote {csv_path(name)}', flush=True)
    rep.add_output(csv_path(name), input='source_states (E)', **exp)


def main():
    print(device, f'| source {SOURCE_STATES} | seed {SEED}', '| FORCE' if os.environ.get('LOKA_FORCE', '0') not in ('', '0') else '',
          '| SMOKE' if SMOKE else '', flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('batch_forgetting', config=config_from_globals(globals())) as rep:
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
        rep.add_data('forgetting counts', FORGETTING)
        X_flat, Y = load_breakthrough_states(SOURCE_STATES)   # (N, 2 * side^2) positions
        n_all = len(X_flat)
        counts = read_counts(FORGETTING, N_SMOKE if SMOKE else n_all)
        if SMOKE:
            X_flat, Y = X_flat[:N_SMOKE], Y[:N_SMOKE]
        assert len(counts) == len(X_flat), f'{FORGETTING}: {len(counts)} rows, E has {len(X_flat)}'
        board_size = board_size_of(X_flat)
        print(f'loaded {len(X_flat)} states, forgetting count 0: {(counts == 0).mean():.1%}, max {counts.max()}', flush=True)

        # cached: hardest first (the cache is only valid for the full file, a smoke run has its own directory)
        if fresh(HARD_ORDER):
            hard = hardest_first(counts)
            HARD_ORDER.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(HARD_ORDER, hard, fmt='%d')
            rep.add_output(HARD_ORDER, method='forgetting hardest first')
        else:
            hard = np.loadtxt(HARD_ORDER, dtype=np.int64)
            assert len(hard) == len(counts), f'{HARD_ORDER} does not match the data, delete it'
        extra_orders = {'forgetting': hard, 'random': np.random.default_rng(TIE_SEED + 1).permutation(len(counts))}

        X_t = torch.as_tensor(X_flat, dtype=torch.float32, device=device)   # features the models train on
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)

        for name in todo:
            run_one(rep, name, EXPERIMENTS[name], X_flat, X_t, Y_t, board_size, extra_orders)


if __name__ == '__main__':
    main()
