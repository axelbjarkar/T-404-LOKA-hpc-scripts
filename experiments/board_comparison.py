"""Board comparison (22 Nov, 'Loka experiments.xlsx'): the 6x6 Breakthrough dataset bt6x6_fens.txt, position-only
features, herding / k-center subsets, CNN and Residual model (Residual sized to the CNN's parameter count).

  BC1 herding + CNN       BC2 k-center + CNN       BC3 herding + Residual       BC4 k-center + Residual

For every split (1%..10% of E, 10 splits) a fresh model is trained and we record the MSE on the training subset
(`train_mse`) and on all of E (`mse`) -> results/<id>.csv with columns ratio, mse, train_mse.
The board size is read from the data (6 here), nothing in this script is 5x5 specific.

Orders (indexes into E) are cached per method, BC1/BC3 share the herding order and BC2/BC4 the k-center order;
LOKA_FORCE=1 recomputes them (and the results). Run a subset with LOKA_EXPERIMENTS="BC1,BC4" (comma separated, wildcards).
The constant-predictor MSE on E (1 - mean(Y)^2) is printed and logged as a baseline: label balance differs per dataset."""
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
SOURCE_STATES  = ROOT / "data" / "game_states" / f"{os.environ.get('LOKA_SOURCE', 'bt6x6_fens')}.txt"
SMOKE          = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")   # tiny run to check the pipeline; own directory
EXPERIMENT_DIR = ROOT / ("data/board_comparison_smoke" if SMOKE else "data/board_comparison") / SOURCE_STATES.stem
RESULTS_DIR    = EXPERIMENT_DIR / "results"     # <id>.csv
ORDERS_DIR     = EXPERIMENT_DIR / "orders"      # <method>.txt: indexes into E

# ---- CONFIG: experiment settings ----
START = 0.01
END = 0.1
STEPS = 10
EPOCHS = 50
BATCH_SIZE = 512
EVAL_BS = 8192
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name
if SMOKE:
    EPOCHS, STEPS = 2, 2

EXPERIMENTS = {   # id -> setup (position-only features for all of them)
    'BC1': dict(method='herding',  model='cnn'),
    'BC2': dict(method='k_center', model='cnn'),
    'BC3': dict(method='herding',  model='residual'),
    'BC4': dict(method='k_center', model='residual'),
}

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
    return RESULTS_DIR / f'{name}.csv'


def train(model, optimizer, loss_fn, X, Y):
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), device=device)   # new shuffle each epoch
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            optimizer.step()


def run_one(rep, name, exp, X_flat, X_t, Y_t, board_size):
    method, kind = exp['method'], exp['model']
    order = torch.as_tensor(get_order(ORDERS_DIR / f'{method}.txt', X_flat, method, END, device, rep), device=device)
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    rows = []

    pbar = tqdm(total=len(splits), desc=name)
    for m, s in enumerate(splits):
        idx = order[:int(s * len(X_t))]
        X_S, Y_S = X_t[idx], Y_t[idx]

        torch.manual_seed(SEED)
        model = build_model(kind, X_S, on_code=False, board_size=board_size).to(device)
        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            info = rep.add_model(name, model, input_shape=list(X_S.shape[1:]), embedding='position_only',
                                 method=method, board_size=board_size)
            print(f'{name}: {kind}, {info["n_params"]:,} parameters', flush=True)

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
    del order


def main():
    print(device, f'| source {SOURCE_STATES}', '| FORCE' if os.environ.get('LOKA_FORCE', '0') not in ('', '0') else '',
          '| SMOKE' if SMOKE else '')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('board_comparison', config=config_from_globals(globals())) as rep:
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
        board_size = board_size_of(X_flat)
        baseline = 1 - float(Y.mean()) ** 2                    # best constant predictor on E
        print(f'loaded {len(X_flat)} states, {board_size}x{board_size} board, '
              f'share of label +1: {(Y == 1).mean():.3f}, constant-predictor MSE on E: {baseline:.4f}', flush=True)
        rep.log_metric('baseline/mse_E_constant_predictor', baseline)

        X_t = torch.as_tensor(X_flat, dtype=torch.float32, device=device)   # features the models train on
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)

        for name in todo:
            run_one(rep, name, EXPERIMENTS[name], X_flat, X_t, Y_t, board_size)


if __name__ == '__main__':
    main()
