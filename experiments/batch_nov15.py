"""Batch of 15 Nov (experiment_plan.xlsx): herding (ECH) and k-center (ECK) subsets of o.txt, 5 embeddings x
2 models (CNN, Residual), each also with an autoencoder bottleneck (.AE): order and train on the 10-d code.

For every split (1%..10% of E, 10 splits) a fresh model is trained and we record the MSE on the training subset
(`train_mse`) and on all of E (`mse`) -> results/<id>.csv with columns ratio, mse, train_mse.

Orders (and autoencoder codes) are computed once and cached; LOKA_FORCE=1 recomputes them (and the results).
Run a subset with LOKA_EXPERIMENTS="ECH1,ECK3.AE,ECH*" (comma separated, shell-style wildcards)."""
import fnmatch
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from common import EMBEDDINGS, build_model, get_codes, get_order, fresh, mse
from loka.data import load_breakthrough_states
from loka.report import ExperimentReport, config_from_globals

# ---- CONFIG: input / output locations ----
ROOT           = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES  = ROOT / "data" / "game_states" / f"{os.environ.get('LOKA_SOURCE', 'o')}.txt"
EXPERIMENT_DIR = ROOT / "data/batch_nov15" / SOURCE_STATES.stem
RESULTS_DIR    = EXPERIMENT_DIR / "results"     # <id>.csv
ORDERS_DIR     = EXPERIMENT_DIR / "orders"      # <method>_<embedding>[_ae].txt: indexes into E
CODES_DIR      = EXPERIMENT_DIR / "codes"       # <embedding>.npy: autoencoder bottleneck of every state in E

# ---- CONFIG: experiment settings ----
START = 0.01
END = 0.1
STEPS = 10
EPOCHS = 50
AE_EPOCHS = 50
BOTTLENECK = 10
BATCH_SIZE = 512
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name
SMOKE = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")   # tiny run to check the pipeline
if SMOKE:
    EPOCHS, AE_EPOCHS, STEPS = 2, 2, 2

EMBEDDING_NAMES = ['position_only', 'piece_count', 'near_goal', 'clear_lane', 'all_three']
MODEL_KINDS = ['cnn', 'residual']
METHODS = {'ECH': 'herding', 'ECK': 'k_center'}   # k-center without S0


def make_experiments():
    """40 experiments: {ECH, ECK}{1..10}[.AE]; 1-5 CNN, 6-10 Residual, embeddings in EMBEDDING_NAMES order."""
    exps = {}
    for prefix, method in METHODS.items():
        for i in range(1, 11):
            for ae in (False, True):
                name = f'{prefix}{i}' + ('.AE' if ae else '')
                exps[name] = dict(method=method, embedding=EMBEDDING_NAMES[(i - 1) % 5],
                                  model=MODEL_KINDS[(i - 1) // 5], ae=ae)
    return exps


EXPERIMENTS = make_experiments()

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


def run_one(rep, name, exp, X_flat, Y_t):
    emb, method, kind, ae = exp['embedding'], exp['method'], exp['model'], exp['ae']
    X_emb = EMBEDDINGS[emb](X_flat)
    if ae:
        X_feat = get_codes(CODES_DIR / f'{emb}.npy', X_emb, device, epochs=AE_EPOCHS, seed=SEED,
                           bottleneck=BOTTLENECK, rep=rep, desc=emb)
        order_file = ORDERS_DIR / f'{method}_{emb}_ae.txt'
    else:
        X_feat = X_emb
        order_file = ORDERS_DIR / f'{method}_{emb}.txt'

    order = torch.as_tensor(get_order(order_file, X_feat, method, END, device, rep), device=device)
    X_t = torch.as_tensor(X_feat, dtype=torch.float32, device=device)   # features the model trains on
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    rows = []

    pbar = tqdm(total=len(splits), desc=name)
    for m, s in enumerate(splits):
        idx = order[:int(s * len(X_t))]
        X_S, Y_S = X_t[idx], Y_t[idx]

        torch.manual_seed(SEED)
        model = build_model(kind, X_S, on_code=ae).to(device)
        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            rep.add_model(name, model, input_shape=list(X_S.shape[1:]), embedding=emb, method=method,
                          autoencoder=ae)

        train(model, optimizer, loss_fn, X_S, Y_S)
        train_mse = mse(model, X_S, Y_S)   # on the states it was trained on
        e_mse = mse(model, X_t, Y_t)       # on all of E
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
    del X_t, order


def main():
    print(device, f'| source {SOURCE_STATES}', '| FORCE' if os.environ.get('LOKA_FORCE', '0') not in ('', '0') else '')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('batch_nov15', config=config_from_globals(globals())) as rep:
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
        X_flat, Y = load_breakthrough_states(SOURCE_STATES)   # (N, 50) positions
        print(f'loaded {len(X_flat)} states from {SOURCE_STATES}')
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)

        for name in todo:
            run_one(rep, name, EXPERIMENTS[name], X_flat, Y_t)


if __name__ == '__main__':
    main()
