"""Active learning (CAL) inside herding / k-center prefixes of o.txt: for each prefix S of the order
(1%..10% of E), start with the first INIT_LABEL_SIZE of S labeled, then run AL_ROUNDS rounds of
train -> CAL_select from the rest of S -> label, and finally train and measure MSE on all of E
(ported from 'CAL_herding_and_kcenter.ipynb').

The herding / k-center orders (indexes into E) and one CSV per method are saved in the experiment
directory. Orders are loaded from there if they exist (delete them to recompute)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states
from loka.geometrical_methods import herding_order, k_center_order
from loka.models import get_breakthrough_cnn
from loka.report import ExperimentReport, config_from_globals
from loka.training_methods import CAL_select

# ---- CONFIG: input / output locations ----
ROOT           = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES  = ROOT / "data/game_states/o.txt"                          # all of E
EXPERIMENT_DIR = ROOT / "data/cal_herding_kcenter" / SOURCE_STATES.stem   # everything this experiment writes, per source file
RESULTS_DIR    = EXPERIMENT_DIR / "results"                               # one CSV per method

# ---- CONFIG: experiment settings ----
START = 0.01
END = 0.1
STEPS = 50
EPOCHS = 50               # epochs per training round
AL_ROUNDS = 20
INIT_LABEL_SIZE = 0.1     # share of S that is labeled before the first round
K_NEAREST = 20
BATCH_SIZE = 512
EVAL_BS = 1024
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ORDERS = {   # name -> function(X) computing the order (only the prefix of length END * |E| is needed)
    'herding':  lambda X: herding_order(X, END, device),
    'k_center': lambda X: k_center_order(X, END, device),
}


def csv_path(name):
    return RESULTS_DIR / f'cal_{name}.csv'


def order_path(name):
    return EXPERIMENT_DIR / f'{name}_order.txt'


def get_order(rep, name, X):
    path = order_path(name)
    if path.exists():
        order = np.loadtxt(path, dtype=np.int64)
        assert len(order) >= int(END * len(X)), f'{path} is shorter than END * |E|'
        print(f'loaded {name} order from {path}')
        rep.add_data(f'{name} order', path)
    else:
        order = ORDERS[name](X)
        np.savetxt(path, order, fmt='%d')
        print(f'wrote {path}')
        rep.add_output(path, input='source_states (E)')
    return order


def train(model, optimizer, loss_fn, X, Y, epochs, pbar, pbar_inc):
    """Train on (X, Y) for `epochs` epochs, advancing pbar by pbar_inc per epoch."""
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)   # new shuffle each epoch
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            optimizer.step()
        pbar.update(pbar_inc)


def total_work(n, n_orders):
    """Total pbar units: every epoch of every training round counts as the size of the labeled set."""
    work = 0
    for s in np.linspace(START, END, num=STEPS, endpoint=True):
        S_size = int(s * n)
        init = int(S_size * INIT_LABEL_SIZE)
        size_Q = (S_size - init) // AL_ROUNDS
        work += sum(init + r * size_Q for r in range(AL_ROUNDS + 1))
    return n_orders * EPOCHS * work


def run_one(rep, name, order, X_t, Y_t, pbar):
    order = torch.as_tensor(order, device=device)
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    mse_values, prc_values = [], []

    for m, s in enumerate(splits):
        S_size = int(s * len(X_t))
        init = int(S_size * INIT_LABEL_SIZE)
        size_Q = (S_size - init) // AL_ROUNDS

        init_idx = order[:init]
        X_label, Y_label = X_t[init_idx], Y_t[init_idx]

        mask = torch.zeros(len(X_t), dtype=torch.bool, device=device)
        mask[order[:S_size]] = True
        mask[init_idx] = False          # pool = S minus the initially labeled
        X_pool, Y_pool = X_t[mask], Y_t[mask]

        torch.manual_seed(SEED)
        model = get_breakthrough_cnn(X_label).to(device)
        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            rep.add_model(f'cal_{name}', model, input_shape=list(X_label.shape[1:]), embedding='positions_only')

        for _ in range(AL_ROUNDS):
            train(model, optimizer, loss_fn, X_label, Y_label, EPOCHS, pbar, len(X_label))   # train first, then pick
            Q_idx = CAL_select(model, X_label, X_pool, size_Q, k_nearest=K_NEAREST)

            X_label = torch.cat([X_label, X_pool[Q_idx]], dim=0)
            Y_label = torch.cat([Y_label, Y_pool[Q_idx]], dim=0)   # oracle step

            keep = torch.ones(len(X_pool), dtype=torch.bool, device=device)
            keep[Q_idx] = False
            X_pool, Y_pool = X_pool[keep], Y_pool[keep]

        # final train on the full labeled set, then evaluate on all of E
        train(model, optimizer, loss_fn, X_label, Y_label, EPOCHS, pbar, len(X_label))

        model.eval()
        sq_err = 0.0
        with torch.no_grad():
            for i in range(0, len(X_t), EVAL_BS):
                sq_err += F.mse_loss(model(X_t[i:i + EVAL_BS]), Y_t[i:i + EVAL_BS], reduction='sum').item()
        mse = sq_err / Y_t.numel()
        mse_values.append(mse)
        prc_values.append(s)
        rep.log_metric(f'cal_{name}/mse_E', mse, step=float(s))

        del model, optimizer, X_label, Y_label, X_pool, Y_pool
        torch.cuda.empty_cache()
        pbar.set_postfix(model=f'{m+1}/{STEPS}', split=f'{s:.2f}', run=name, mse=f'{mse:.6f}')

    pd.DataFrame({'ratio': prc_values, 'mse': mse_values}).to_csv(csv_path(name), index=False)
    print(f'wrote {csv_path(name)}')
    rep.add_output(csv_path(name), input='source_states (E)', method=name)


def main():
    print(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('cal_herding_kcenter', config=config_from_globals(globals())) as rep:
        todo = []
        for name in ORDERS:
            if csv_path(name).exists():
                print(f'skipping {name}: {csv_path(name)} already exists')
                rep.add_skipped(csv_path(name))
            else:
                todo.append(name)
        if not todo:
            return

        rep.add_data('source_states (E)', SOURCE_STATES)
        X, Y = load_breakthrough_states(SOURCE_STATES)
        print(f'loaded {len(X)} states from {SOURCE_STATES}')

        orders = {name: get_order(rep, name, X) for name in todo}

        X_t = torch.FloatTensor(X).to(device)
        Y_t = torch.FloatTensor(Y).to(device)
        pbar = tqdm(total=total_work(len(X_t), len(todo)), desc='training',
                    bar_format="{l_bar}{bar} {percentage:3.0f}% [{elapsed}<{remaining}]")
        for name in todo:
            run_one(rep, name, orders[name], X_t, Y_t, pbar)
        pbar.close()


if __name__ == '__main__':
    main()
