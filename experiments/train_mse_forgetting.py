"""Train the CNN on growing prefixes of o.txt ordered by forgetting count (highest first), MSE on all of E
(ported from 'M2/Forgetting_perf.ipynb')."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_forgetting_states
from loka.models import BreakthroughCNN

# ---- CONFIG: input / output locations ----
ROOT        = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_FILE = ROOT / "data/forgetting_counts/forgetting_o.txt"   # all of E, with a forgetting count per state
RESULTS_DIR = ROOT / "data/ordered-data/results/mse_all"
CSV_FILE    = RESULTS_DIR / "forgetting.csv"

# ---- CONFIG: experiment settings ----
START = 0.01
END = 0.1
STEPS = 50
EPOCHS = 50
BATCH_SIZE = 512
EVAL_BS = 1024
LR = 0.001
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(device)
    if CSV_FILE.exists():
        print(f'skipping: {CSV_FILE} already exists')
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X, Y, C = load_forgetting_states(SOURCE_FILE)
    print(f'loaded {len(X)} states from {SOURCE_FILE}')

    order = torch.topk(torch.tensor(C), len(C)).indices.numpy()  # most forgotten first

    X_t = torch.FloatTensor(X).to(device)
    Y_t = torch.FloatTensor(Y).to(device)

    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    mse_values = []
    prc_values = []
    pbar = tqdm(total=STEPS * EPOCHS, desc='training forgetting')

    for m, ratio in enumerate(splits):
        train_idx = order[:int(ratio * len(X))]  # first `ratio` of the data, in order

        torch.manual_seed(SEED)  # same init weights each time
        model = BreakthroughCNN().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        X_train_t = torch.FloatTensor(X[train_idx]).to(device)
        Y_train_t = torch.FloatTensor(Y[train_idx]).to(device)

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
            for i in range(0, len(X_t), EVAL_BS):
                xb, yb = X_t[i:i + EVAL_BS], Y_t[i:i + EVAL_BS]
                sq_err += F.mse_loss(model(xb), yb, reduction='sum').item()
                n += yb.numel()
        mse = sq_err / n
        mse_values.append(mse)
        prc_values.append(ratio)

        # free GPU memory before the next model
        del model, optimizer, X_train_t, Y_train_t
        torch.cuda.empty_cache()

        pbar.set_postfix(model=f'{m+1}/{STEPS}', ratio=f'{ratio:.2f}',
                         loss=f'{loss.item():.4f}', mse=f'{mse:.6f}')

    pbar.close()

    # one row per split: training ratio and MSE
    pd.DataFrame({'ratio': prc_values, 'mse': mse_values}).to_csv(CSV_FILE, index=False)
    print(f'wrote {CSV_FILE}')


if __name__ == '__main__':
    main()
