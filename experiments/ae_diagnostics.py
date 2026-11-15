"""Diagnostics for the .AE rows of batch_nov15: is the weak result caused by the code, the model on the code,
or the ordering? Reads the cached codes and orders written by batch_nov15.py (trains no autoencoder).

Per embedding (skipped if its codes file is missing) it records:
  stats   code scale, dead dimensions, effective rank (PCA participation ratio) of the 10-d code
  knn     validation MSE of a k-nearest-neighbour regressor on the codes vs on the raw features
          (kNN good but the 1D models bad -> the model on the code is the problem; kNN also bad -> the code lost information)
  train   a model trained on the first RATIO of an order, reported as train MSE and MSE on E:
            codes_raw     1D CNN on the codes, herding-on-codes order            (what the .AE rows do)
            codes_std     same, but the codes are standardized first            (code scale)
            ae_order_2d   2D CNN on the original features, herding-on-codes order  (is the ordering the problem?)
            raw_order_2d  2D CNN on the original features, herding order on the features (reference, the non-AE row)

Results: <batch dir>/diagnostics/diagnostics.csv, and a summary table is printed.
Settings by environment: LOKA_EMBEDDINGS="near_goal,clear_lane" (default all), LOKA_RATIO (default 0.05),
LOKA_BATCH_DIR (default $LOKA_ROOT/data/batch_nov15/<source>), LOKA_SMOKE=1 for a tiny run."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import EMBEDDINGS, build_model, mse
from loka.data import load_breakthrough_states

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data" / "game_states" / f"{os.environ.get('LOKA_SOURCE', 'o')}.txt"
BATCH_DIR     = Path(os.environ.get("LOKA_BATCH_DIR", ROOT / "data/batch_nov15" / SOURCE_STATES.stem))
CODES_DIR     = BATCH_DIR / "codes"
ORDERS_DIR    = BATCH_DIR / "orders"
OUT_DIR       = BATCH_DIR / "diagnostics"

# ---- CONFIG: settings ----
RATIO = float(os.environ.get("LOKA_RATIO", "0.05"))   # share of E the diagnostic models train on
EMBEDDING_NAMES = [e for e in os.environ.get("LOKA_EMBEDDINGS", ",".join(EMBEDDINGS)).split(",") if e]
N_PROBE = 100_000      # states sampled for the kNN probe (80% reference, 20% validation)
K_NEAREST = 10
EPOCHS = 50
BATCH_SIZE = 512
LR = 0.001
SEED = 42
SMOKE = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")
if SMOKE:
    N_PROBE, EPOCHS = 4000, 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def code_stats(codes):
    std = codes.std(axis=0)
    eig = np.clip(np.linalg.eigvalsh(np.cov(codes, rowvar=False)), 0, None)
    return {
        'dim': codes.shape[1],
        'dead_dims': int((std < 1e-6).sum()),                       # constant (e.g. always-0 after ReLU)
        'min_std': float(std.min()), 'max_std': float(std.max()),   # very different scales -> standardize
        'mean_abs': float(np.abs(codes).mean()),
        'effective_rank': float(eig.sum() ** 2 / (eig ** 2).sum()),  # 10 = all dims used equally, 1 = one direction
    }


@torch.no_grad()
def knn_mse(X_ref, Y_ref, X_val, Y_val, k=K_NEAREST, chunk=2048):
    """MSE of predicting each validation outcome by the mean outcome of its k nearest reference states."""
    sq_err = 0.0
    for s in range(0, len(X_val), chunk):
        idx = torch.cdist(X_val[s:s + chunk], X_ref).topk(k, largest=False).indices
        sq_err += ((Y_ref[idx].mean(dim=1) - Y_val[s:s + chunk]) ** 2).sum().item()
    return sq_err / len(X_val)


def train_eval(kind, on_code, X_feat, Y, order):
    """Train a fresh model on the first RATIO of `order`; return (train MSE, MSE on all of E)."""
    idx = order[:int(RATIO * len(X_feat))]
    X_S, Y_S = X_feat[idx], Y[idx]
    torch.manual_seed(SEED)
    model = build_model(kind, X_S, on_code=on_code).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X_S), device=device)
        for i in range(0, len(X_S), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X_S[b]), Y_S[b]).backward()
            optimizer.step()
    result = mse(model, X_S, Y_S), mse(model, X_feat, Y)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def load_order(name):
    path = ORDERS_DIR / name
    if not path.exists():
        print(f'  (no {path.name}: skipping the variants that need it)', flush=True)
        return None
    order = np.loadtxt(path, dtype=np.int64)
    return torch.as_tensor(order, device=device)


def main():
    print(device, f'| batch dir {BATCH_DIR} | ratio {RATIO}', '| SMOKE' if SMOKE else '', flush=True)
    X_flat, Y_np = load_breakthrough_states(SOURCE_STATES)
    Y = torch.as_tensor(Y_np, dtype=torch.float32, device=device)
    print(f'loaded {len(X_flat)} states', flush=True)

    rng = np.random.default_rng(SEED)
    sample = rng.permutation(len(X_flat))[:N_PROBE]
    n_ref = int(0.8 * len(sample))
    ref, val = torch.as_tensor(sample[:n_ref], device=device), torch.as_tensor(sample[n_ref:], device=device)

    rows = []

    def add(emb, check, variant, **metrics):
        for metric, value in metrics.items():
            rows.append({'embedding': emb, 'check': check, 'variant': variant, 'metric': metric, 'value': value})

    for emb in EMBEDDING_NAMES:
        codes_path = CODES_DIR / f'{emb}.npy'
        if not codes_path.exists():
            print(f'== {emb}: no codes at {codes_path}, skipping', flush=True)
            continue
        print(f'== {emb}', flush=True)
        codes = np.load(codes_path)
        X_emb = torch.as_tensor(EMBEDDINGS[emb](X_flat), dtype=torch.float32, device=device)
        C = torch.as_tensor(codes, dtype=torch.float32, device=device)

        add(emb, 'stats', 'codes', **code_stats(codes))

        # kNN: how much does the code tell about the outcome locally, compared with the original features
        add(emb, 'knn', 'codes', val_mse=knn_mse(C[ref], Y[ref], C[val], Y[val]))
        add(emb, 'knn', 'raw_features', val_mse=knn_mse(X_emb[ref], Y[ref], X_emb[val], Y[val]))
        print('  kNN done', flush=True)

        mu, sd = C.mean(dim=0), C.std(dim=0)
        C_std = (C - mu) / (sd + 1e-8)   # dead dims -> 0

        ae_order = load_order(f'herding_{emb}_ae.txt')
        raw_order = load_order(f'herding_{emb}.txt')
        if ae_order is not None:
            for variant, kind, on_code, X_feat, order in [
                ('codes_raw',   'cnn', True,  C,     ae_order),
                ('codes_std',   'cnn', True,  C_std, ae_order),
                ('ae_order_2d', 'cnn', False, X_emb, ae_order),
            ]:
                tr, e = train_eval(kind, on_code, X_feat, Y, order)
                add(emb, 'train', variant, train_mse=tr, mse_E=e)
                print(f'  {variant}: train {tr:.4f}  E {e:.4f}', flush=True)
        if raw_order is not None:
            tr, e = train_eval('cnn', False, X_emb, Y, raw_order)
            add(emb, 'train', 'raw_order_2d', train_mse=tr, mse_E=e)
            print(f'  raw_order_2d: train {tr:.4f}  E {e:.4f}', flush=True)

        del X_emb, C, C_std
        torch.cuda.empty_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'diagnostics.csv', index=False)
    print(f'\nwrote {OUT_DIR / "diagnostics.csv"}')
    if df.empty:
        return
    pd.set_option('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.4f}'.format)
    for check in ('stats', 'knn', 'train'):
        part = df[df.check == check]
        if len(part):
            print(f'\n[{check}]')
            print(part.pivot_table(index=['embedding', 'variant'], columns='metric', values='value', sort=False))


if __name__ == '__main__':
    main()
