"""Train the CNN on growing prefixes of each ordered file, MSE on all of E
(ported from 'Training on basic embeddings.ipynb' and 'More_Data_Smaller_Subsets.ipynb')."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states
from loka.report import ExperimentReport, config_from_globals
from loka.models import get_breakthrough_cnn
from loka.embeddings import with_counts, with_near_goal, with_counts_and_near_goal

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/o.txt"                       # all of E, also the "random" ordering
ORDERED_DIR   = ROOT / "data/ordered-data"                # positions-only orderings
EMB_DIR       = ROOT / "data/ordered-data/embeddings"     # embedding orderings
RESULTS_DIR   = ROOT / "data/ordered-data/results/mse_all"

# ---- CONFIG: experiment settings ----
START = 0.01
END = 0.1
STEPS = 50
EPOCHS = 50
BATCH_SIZE = 512
EVAL_BS = 1024
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"   # torch.optim name
LOSS = "MSELoss"     # torch.nn name

# (csv name, ordered file); names match the CSVs Aggregation already plots
POSITION_INPUTS = [
    ("random",               SOURCE_STATES),
    ("herding",              ORDERED_DIR / "herding_25p.txt"),
    ("k_center_no_S0",       ORDERED_DIR / "k_center_no_S0_25p.txt"),
    ("k_center_rand_S0_25p", ORDERED_DIR / "k_center_rand_S0_25p.txt"),
]
INCLUDE_EMBEDDING_INPUTS = True  # also train on every .txt in EMB_DIR

EMB_METHODS = [with_counts, with_near_goal, with_counts_and_near_goal]

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def positions_only(X):
    return X.reshape(len(X), -1)


def emb_for(name):
    base = name.rsplit("_", 1)[0]  # drop the "_10p" suffix
    # longest name first so "with_counts" can't shadow "with_counts_and_near_goal"
    for emb in sorted(EMB_METHODS, key=lambda e: len(e.__name__), reverse=True):
        if base.endswith(emb.__name__):
            return emb
    raise ValueError(f"no embedding method matches {name}")


def collect_inputs():
    inputs = [(name, path, positions_only) for name, path in POSITION_INPUTS]
    if INCLUDE_EMBEDDING_INPUTS and EMB_DIR.is_dir():
        for path in sorted(EMB_DIR.glob("*.txt")):
            inputs.append((path.stem, path, emb_for(path.stem)))
    return inputs


def train_one(rep, name, path, emb, X_all, Y_all_t, csv_file):
    total_data_amount = X_all.shape[0]
    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    mse_values = []
    prc_values = []
    pbar = tqdm(total=STEPS * EPOCHS, desc=f'training {name}')

    # load the ordered subset and apply the embedding it was selected with
    X, Y = load_breakthrough_states(path)
    X = emb(X)
    rep.add_data(name, path)

    # full dataset through the same embedding, for evaluation
    X_all_t = torch.FloatTensor(emb(X_all)).to(device)

    for m, ratio in enumerate(splits):
        n_train = int(total_data_amount * ratio)
        X_train, Y_train = X[:n_train], Y[:n_train]  # first `ratio` of the data, in order

        model = get_breakthrough_cnn(X_train).to(device)  # channels sized to the embedding

        optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
        loss_fn = getattr(nn, LOSS)()
        if m == 0:
            rep.add_model(name, model, input_shape=list(X_train.shape[1:]), embedding=emb.__name__)

        X_train_t = torch.FloatTensor(X_train).to(device)
        Y_train_t = torch.FloatTensor(Y_train).to(device)

        for epoch in range(EPOCHS):
            perm = torch.randperm(len(X_train_t), device=device)  # new shuffle each epoch

            for i in range(0, len(X_train_t), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb, yb = X_train_t[idx], Y_train_t[idx]

                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
            pbar.update(1)

        # batched evaluation on all of E
        sq_err, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_all_t), EVAL_BS):
                xb, yb = X_all_t[i:i + EVAL_BS], Y_all_t[i:i + EVAL_BS]
                sq_err += F.mse_loss(model(xb), yb, reduction='sum').item()
                n += yb.numel()
        mse = sq_err / n
        mse_values.append(mse)
        rep.log_metric(f'{name}/mse_E', mse, step=float(ratio))
        prc_values.append(ratio)

        # free GPU memory before the next model
        del model, optimizer, X_train_t, Y_train_t
        torch.cuda.empty_cache()

        pbar.set_postfix(model=f'{m+1}/{STEPS}', ratio=f'{ratio:.2f}',
                         loss=f'{loss.item():.4f}', mse=f'{mse:.6f}')

    pbar.close()
    del X_all_t
    torch.cuda.empty_cache()

    # one row per split: training ratio and MSE
    pd.DataFrame({'ratio': prc_values, 'mse': mse_values}).to_csv(csv_file, index=False)
    print(f'wrote {csv_file}')
    rep.add_output(csv_file, input=name)


def main():
    print(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('train_mse_all', config=config_from_globals(globals())) as rep:
        rep.add_data('source_states (E)', SOURCE_STATES)
        X_all, Y_all = load_breakthrough_states(SOURCE_STATES, two_channel=True)
        Y_all_t = torch.FloatTensor(Y_all).to(device)
        print(f'loaded {len(X_all)} states from {SOURCE_STATES}')

        for name, path, emb in collect_inputs():
            csv_file = RESULTS_DIR / f'{name}.csv'
            if csv_file.exists():
                print(f'skipping {name}: {csv_file} already exists')
                rep.add_skipped(csv_file)
                continue
            if not Path(path).exists():
                print(f'Warning: skipping {name}: input not found: {path}')
                continue
            train_one(rep, name, path, emb, X_all, Y_all_t, csv_file)


if __name__ == '__main__':
    main()
