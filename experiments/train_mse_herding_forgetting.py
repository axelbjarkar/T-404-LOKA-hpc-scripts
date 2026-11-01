"""Train the CNN on growing prefixes, first on herding order and then on forgetting order (highest
count first), for several epoch splits. MSE on all of E (ported from 'M2/Herding_and_Forgetting_events.ipynb')."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states, load_forgetting_states
from loka.models import BreakthroughCNN
from loka.report import ExperimentReport, config_from_globals

# ---- CONFIG: input / output locations ----
ROOT            = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
HERDING_FILE    = ROOT / "data/ordered-data/herding_e_25p.txt"   # from order_herding_e_25p.py
FORGETTING_FILE = ROOT / "data/forgetting_counts/forgetting_e.txt"   # all of E, with a forgetting count per state
RESULTS_DIR     = ROOT / "data/forgetting_counts/herding_and_forgetting_e"

# ---- CONFIG: experiment settings ----
HERDING_RATIOS = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2]   # share of the epochs spent on herding data, the rest on forgetting data
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


def train(model, optimizer, loss_fn, X, Y, epochs, pbar, n_points):
    X_t, Y_t = X[:n_points], Y[:n_points]  # first n_points of the data, in order

    for _ in range(epochs):
        perm = torch.randperm(len(X_t), device=device)  # new shuffle each epoch

        for i in range(0, len(X_t), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = X_t[idx], Y_t[idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        pbar.update(n_points)  # epochs cost more on larger prefixes


def run_ratio(name, herding_ratio, X_h, Y_h, X_f, Y_f, X_all, Y_all, csv_file, rep):
    herding_epochs = round(herding_ratio * EPOCHS)
    forgetting_epochs = EPOCHS - herding_epochs

    splits = np.linspace(START, END, num=STEPS, endpoint=True)
    n_points_per_split = [int(s * len(X_all)) for s in splits]
    mse_values = []
    prc_values = []
    pbar = tqdm(total=EPOCHS * sum(n_points_per_split), desc=f'training {name}',
                bar_format="{l_bar}{bar} {percentage:3.0f}% [{elapsed}<{remaining}]{postfix}")

    for m, (s, n_points) in enumerate(zip(splits, n_points_per_split)):
        torch.manual_seed(SEED)  # same init weights each time
        model = BreakthroughCNN().to(device)
        if m == 0:
            rep.add_model(name, model, input_shape=list(X_all.shape[1:]), embedding='positions_only')

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        # train first with the herding data, then with the forgetting data
        train(model, optimizer, loss_fn, X_h, Y_h, herding_epochs, pbar, n_points)
        train(model, optimizer, loss_fn, X_f, Y_f, forgetting_epochs, pbar, n_points)

        # batched evaluation on all of E
        sq_err, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_all), EVAL_BS):
                xb, yb = X_all[i:i + EVAL_BS], Y_all[i:i + EVAL_BS]
                sq_err += F.mse_loss(model(xb), yb, reduction='sum').item()
                n += yb.numel()
        mse = sq_err / n
        mse_values.append(mse)
        prc_values.append(s)
        rep.log_metric(f'{name}/mse', mse, step=float(s))

        # free GPU memory before the next model
        del model, optimizer
        torch.cuda.empty_cache()

        pbar.set_postfix(model=f'{m+1}/{STEPS}', ratio=f'{s:.2f}', mse=f'{mse:.6f}')

    pbar.close()

    # one row per split: training ratio and MSE
    pd.DataFrame({'ratio': prc_values, 'mse': mse_values}).to_csv(csv_file, index=False)
    print(f'wrote {csv_file}')
    rep.add_output(csv_file, herding_epochs=herding_epochs, forgetting_epochs=forgetting_epochs)


def main():
    print(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ExperimentReport('train_mse_herding_forgetting', config=config_from_globals(globals())) as rep:
        rep.add_data('herding', HERDING_FILE)
        rep.add_data('forgetting (E)', FORGETTING_FILE)

        X_h, Y_h = load_breakthrough_states(HERDING_FILE)
        X_all, Y_all, C = load_forgetting_states(FORGETTING_FILE)
        print(f'loaded {len(X_h)} herding states from {HERDING_FILE}')
        print(f'loaded {len(X_all)} states from {FORGETTING_FILE}')

        order = torch.topk(torch.tensor(C), len(C)).indices.numpy()  # most forgotten first

        X_h = torch.FloatTensor(X_h).to(device)
        Y_h = torch.FloatTensor(Y_h).to(device)
        X_all = torch.FloatTensor(X_all).to(device)
        Y_all = torch.FloatTensor(Y_all).to(device)
        X_f, Y_f = X_all[order], Y_all[order]

        for herding_ratio in HERDING_RATIOS:
            h = round(herding_ratio * 100)
            name = f'herding_{h}_forgetting_{100 - h}'
            csv_file = RESULTS_DIR / f'{name}.csv'
            if csv_file.exists():
                print(f'skipping {name}: {csv_file} already exists')
                rep.add_skipped(csv_file)
                continue
            run_ratio(name, herding_ratio, X_h, Y_h, X_f, Y_f, X_all, Y_all, csv_file, rep)


if __name__ == '__main__':
    main()
