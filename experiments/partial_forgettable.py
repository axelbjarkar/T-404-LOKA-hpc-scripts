"""Train the CNN on herding prefixes of e.txt where the last x% of the slots are forgettable events,
MSE on all of E (ported from 'Partial Forgettable subset.ipynb').

The forgetting counts and the herding order (indexes into E) are saved in the experiment directory and
loaded from there on later runs if they exist (delete them to recompute). Subsets are built in memory and not saved."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states, load_forgetting_states, save_forgetting_states
from loka.geometrical_methods import herding_order
from loka.models import get_breakthrough_cnn
from loka.report import ExperimentReport, config_from_globals
from loka.training_methods import ForgettingScores, make_forgettable_subset

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/e.txt"                       # all of E
EXPERIMENT_DIR = ROOT / "data/partial_forgettable"                    # everything this experiment writes
RESULTS_DIR   = EXPERIMENT_DIR / "results"                            # per-ratio CSVs
COUNTS_FILE   = EXPERIMENT_DIR / "forgetting_counts.txt"              # E with a forgetting count after each state (same format as forgetting_e.txt)
ORDER_FILE    = EXPERIMENT_DIR / "herding_order.txt"                  # herding order: indexes into E

# ---- CONFIG: experiment settings ----
FORGETTABLE_RATIOS = [0.01, 0.05, 0.1, 0.2, 0.3]   # share of each subset (from the end) made forgettable
START = 0.01
END = 0.1
STEPS = 50
EPOCHS = 50
FORGETTING_EPOCHS = 200   # epochs for the forgetting-count run on all of E
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


def csv_path(ratio):
    return RESULTS_DIR / f'forgettable_{round(ratio * 100)}p.csv'


def forgetting_counts(rep, X, Y):
    model = get_breakthrough_cnn(X)
    rep.add_model('forgetting', model, input_shape=list(X.shape[1:]), embedding='positions_only')
    forgetter = ForgettingScores(model, seed=SEED)
    pbar = tqdm(total=FORGETTING_EPOCHS, desc='forgetting counts')
    forgetter.train(X=X, Y=Y, epochs=FORGETTING_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    optimizer=OPTIMIZER, device=device, pbar=pbar)
    pbar.close()
    return forgetter.getForgettingCounts()


def train(model, optimizer, loss_fn, X, Y, epochs, pbar, pbar_inc):
    """Train on (X, Y) for `epochs` epochs, advancing pbar by pbar_inc per epoch. Returns the last batch loss."""
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=device)   # new shuffle each epoch
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X[idx]), Y[idx])
            loss.backward()
            optimizer.step()
        pbar.update(pbar_inc)
    return loss.item()


def total_work(n, n_ratios):
    """Total pbar units: every epoch of every model counts as its number of training points."""
    return n_ratios * EPOCHS * sum(int(s * n) for s in np.linspace(START, END, num=STEPS, endpoint=True))


def train_one(rep, ratio, X_t, Y_t, C, order, pbar):
    name = f'forgettable_{round(ratio * 100)}p'
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    mse_values, prc_values = [], []

    for m, s in enumerate(splits):
        S_size = int(s * len(X_t))
        forgettable_size = int(S_size * ratio)
        X_S, Y_S = make_forgettable_subset(X_t, Y_t, C, order, S_size, forgettable_size)

        torch.manual_seed(SEED)
        model = get_breakthrough_cnn(X_S).to(device)
        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            rep.add_model(name, model, input_shape=list(X_S.shape[1:]), embedding='positions_only')

        loss = train(model, optimizer, loss_fn, X_S, Y_S, EPOCHS, pbar, pbar_inc=len(X_S))

        # batched evaluation on all of E
        model.eval()
        sq_err = 0.0
        with torch.no_grad():
            for i in range(0, len(X_t), EVAL_BS):
                sq_err += F.mse_loss(model(X_t[i:i + EVAL_BS]), Y_t[i:i + EVAL_BS], reduction='sum').item()
        mse = sq_err / Y_t.numel()
        mse_values.append(mse)
        prc_values.append(s)
        rep.log_metric(f'{name}/mse_E', mse, step=float(s))

        del model, optimizer, X_S, Y_S
        torch.cuda.empty_cache()
        pbar.set_postfix(model=f'{m+1}/{STEPS}', split=f'{s:.2f}', run=name, loss=f'{loss:.4f}', mse=f'{mse:.6f}')

    pd.DataFrame({'ratio': prc_values, 'mse': mse_values}).to_csv(csv_path(ratio), index=False)
    print(f'wrote {csv_path(ratio)}')
    rep.add_output(csv_path(ratio), input='source_states (E)', forgettable_ratio=ratio)


def main():
    print(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('partial_forgettable', config=config_from_globals(globals())) as rep:
        todo = []
        for ratio in FORGETTABLE_RATIOS:
            if csv_path(ratio).exists():
                print(f'skipping {ratio}: {csv_path(ratio)} already exists')
                rep.add_skipped(csv_path(ratio))
            else:
                todo.append(ratio)
        if not todo:
            return

        rep.add_data('source_states (E)', SOURCE_STATES)
        X, Y = load_breakthrough_states(SOURCE_STATES)
        print(f'loaded {len(X)} states from {SOURCE_STATES}')

        if COUNTS_FILE.exists():
            X_c, Y_c, C = load_forgetting_states(COUNTS_FILE)
            assert np.array_equal(X_c, X) and np.array_equal(Y_c, Y), f'{COUNTS_FILE} does not match {SOURCE_STATES}'
            C = torch.as_tensor(C, dtype=torch.int32, device=device)
            print(f'loaded forgetting counts from {COUNTS_FILE}')
            rep.add_data('forgetting counts', COUNTS_FILE)
        else:
            C = forgetting_counts(rep, X, Y)
            save_forgetting_states(COUNTS_FILE, X, Y, C.cpu().numpy())
            print(f'wrote {COUNTS_FILE}')
            rep.add_output(COUNTS_FILE, input='source_states (E)')
        print(f'forgettable events: {(C != 0).float().mean().item():.1%} of E')

        if ORDER_FILE.exists():
            order = np.loadtxt(ORDER_FILE, dtype=np.int64)
            assert len(order) >= int(END * len(X)), f'{ORDER_FILE} is shorter than END * |E|'
            print(f'loaded herding order from {ORDER_FILE}')
            rep.add_data('herding order', ORDER_FILE)
        else:
            order = herding_order(X, END, device)   # only the largest subset's prefix is needed
            np.savetxt(ORDER_FILE, order, fmt='%d')
            print(f'wrote {ORDER_FILE}')
            rep.add_output(ORDER_FILE, input='source_states (E)')

        X_t = torch.FloatTensor(X).to(device)
        Y_t = torch.FloatTensor(Y).to(device)
        pbar = tqdm(total=total_work(len(X_t), len(todo)), desc='training',
                    bar_format="{l_bar}{bar} {percentage:3.0f}% [{elapsed}<{remaining}]")
        for ratio in todo:
            train_one(rep, ratio, X_t, Y_t, C, order, pbar)
        pbar.close()


if __name__ == '__main__':
    main()
