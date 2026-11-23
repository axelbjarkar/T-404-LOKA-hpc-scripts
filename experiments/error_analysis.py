"""Where does the error on E sit? Train a model on the first RATIO of a cached order (as the ECH/ECK position-only rows do),
compute the squared error of every state of E and group it by the state's forgetting count.

If a small set of high-forgetting states holds most of the error, mixing hard states into the subset has something to gain;
if the error is spread out, it does not. Run for the CNN and the Residual model (LOKA_MODELS) to see whether the model that
fits the training set better (Residual) has its error in different states.

Inputs (E = $LOKA_ROOT/data/game_states/<source>.txt, default o):
  forgetting counts   $LOKA_ROOT/data/forgetting_counts/forgetting_<source>.txt   one row per state of E, same order, count = last field
  order               <orders dir>/<method>_position_only.txt                     cached by batch_nov15 (computed if missing)
Results in $LOKA_ROOT/data/error_analysis[_smoke]/<source>/:
  summary.csv      model, seed, mse_E, train_mse, spearman(count, error), error share of the top 1% / 5% states by count
  bins.csv         per model/seed and forgetting-count bin (and in S / not in S): state share, mean error, error share
  gain_curve.csv   cumulative state share -> cumulative error share, states sorted by count (high first) and by true error (oracle)
Settings by environment: LOKA_MODELS="cnn,residual", LOKA_METHOD=herding|k_center, LOKA_RATIO=0.1, LOKA_SEEDS=3,
LOKA_ORDERS_DIR, LOKA_SMOKE=1 (first 100k states, 2 epochs, own directory and orders).

LOKA_CAL=1 additionally checks whether a label-free CAL score finds the hard states. Pool = E minus S; every pool state gets the mean
|prediction(neighbour) - prediction(state)| over its LOKA_CAL_K (20) nearest states in S (labels of the pool are never used), with distances in
LOKA_CAL_SPACES="raw,penultimate" (raw positions / inputs of the model's last Linear). cal_summary.csv and cal_gain_curve.csv report, on the
pool only, the error share held by the top 1% / 5% states for each sorter (cal_*, forgetting count and true error as oracles, random)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import board_size_of, build_model, get_order, mse
from loka.data import load_breakthrough_states
from loka.training_methods import CAL_scores

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE        = os.environ.get("LOKA_SOURCE", "o")
SOURCE_STATES = ROOT / "data" / "game_states" / f"{SOURCE}.txt"
FORGETTING    = ROOT / "data" / "forgetting_counts" / f"forgetting_{SOURCE}.txt"
SMOKE         = os.environ.get("LOKA_SMOKE", "0") not in ("", "0")
OUT_DIR       = ROOT / ("data/error_analysis_smoke" if SMOKE else "data/error_analysis") / SOURCE
ORDERS_DIR    = Path(os.environ.get("LOKA_ORDERS_DIR", OUT_DIR / "orders" if SMOKE else ROOT / "data/batch_nov15" / SOURCE / "orders"))

# ---- CONFIG: settings (training as in batch_nov15) ----
METHOD  = os.environ.get("LOKA_METHOD", "herding")
MODELS  = [m for m in os.environ.get("LOKA_MODELS", "cnn,residual").split(",") if m]
RATIO   = float(os.environ.get("LOKA_RATIO", "0.1"))
N_SEEDS = int(os.environ.get("LOKA_SEEDS", "3"))
EPOCHS = 50
BATCH_SIZE = 512
EVAL_BS = 8192
LR = 0.001
SEED = 42
N_SMOKE = 100_000
CAL = os.environ.get("LOKA_CAL", "0") not in ("", "0")
CAL_K = int(os.environ.get("LOKA_CAL_K", "20"))
CAL_SPACES = [x for x in os.environ.get("LOKA_CAL_SPACES", "raw,penultimate").split(",") if x]
HARD = 11             # forgetting count that counts as "hard" for precision / recall
if SMOKE:
    EPOCHS, N_SEEDS = 2, 1

BIN_EDGES = [0, 1, 2, 3, 6, 11, np.inf]                       # counts 0 | 1 | 2 | 3-5 | 6-10 | 11+
BIN_NAMES = ['0', '1', '2', '3-5', '6-10', '11+']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_counts(path, n):
    with open(path) as f:
        counts = np.fromiter((int(line.rsplit(None, 1)[-1]) for line in f), dtype=np.int64, count=n if n else -1)
    return counts


def train(model, X, Y):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), device=device)
        for i in range(0, len(X), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss_fn(model(X[idx]), Y[idx]).backward()
            optimizer.step()


@torch.no_grad()
def sq_errors(model, X, Y):
    model.eval()
    return torch.cat([(model(X[i:i + EVAL_BS]) - Y[i:i + EVAL_BS]) ** 2 for i in range(0, len(X), EVAL_BS)]).cpu().numpy().astype(np.float64)


def share_of_top(err, counts, frac):
    """Share of the total error held by the `frac` states with the highest forgetting count (ties broken by index)."""
    k = max(1, int(frac * len(err)))
    top = np.argsort(-counts, kind='stable')[:k]
    return err[top].sum() / err.sum()


def bin_rows(err, counts, in_s, model, seed):
    total, n = err.sum(), len(err)
    rows = []
    bin_id = np.digitize(counts, BIN_EDGES[1:-1], right=False)
    for name_i, name in enumerate(BIN_NAMES):
        for subset, mask_s in (('all', np.ones(n, bool)), ('in_S', in_s), ('not_in_S', ~in_s)):
            m = (bin_id == name_i) & mask_s
            rows.append(dict(model=model, seed=seed, bin=name, subset=subset, states=int(m.sum()), state_share=m.sum() / n,
                             mean_error=float(err[m].mean()) if m.any() else np.nan, error_share=err[m].sum() / total))
    return rows


def gain_rows(err, counts, model, seed, points=1000):
    n = len(err)
    pos = np.unique(np.linspace(0, n, points + 1).astype(int))[1:]
    rows = []
    for name, order in (('forgetting', np.argsort(-counts, kind='stable')), ('oracle', np.argsort(-err, kind='stable'))):
        cum = np.cumsum(err[order]) / err.sum()
        rows += [dict(model=model, seed=seed, sorted_by=name, state_share=p / n, error_share=cum[p - 1]) for p in pos]
    return rows


def cal_rows(err, counts, in_s, scores_by_space, model, seed, rng):
    """Error share of the top 1% / 5% of the pool (E minus S) per sorter, precision / recall against count >= HARD, gain curves."""
    pool = ~in_s
    e, c = err[pool], counts[pool]
    sorters = {f'cal_{space}': sc for space, sc in scores_by_space.items()}
    sorters.update({'forgetting (oracle)': c.astype(np.float64), 'true error (oracle)': e, 'random': rng.random(len(e))})
    hard = c >= HARD
    rows, curves = [], []
    n = len(e)
    pos = np.unique(np.linspace(0, n, 1001).astype(int))[1:]
    for name, score in sorters.items():
        order = np.argsort(-score, kind='stable')
        cum = np.cumsum(e[order]) / e.sum()
        row = dict(model=model, seed=seed, sorter=name, pool_states=n, pool_mse=e.mean(),
                   spearman=pd.Series(score).corr(pd.Series(e), method='spearman'))
        for frac in (0.01, 0.05):
            top = order[:max(1, int(frac * n))]
            row[f'top{int(frac * 100)}_error_share'] = cum[len(top) - 1]
            row[f'top{int(frac * 100)}_precision_hard'] = hard[top].mean()
            row[f'top{int(frac * 100)}_recall_hard'] = hard[top].sum() / max(1, hard.sum())
        rows.append(row)
        curves += [dict(model=model, seed=seed, sorted_by=name, state_share=p / n, error_share=cum[p - 1]) for p in pos]
    return rows, curves


def main():
    print(device, f'| {METHOD} {RATIO:.0%} | models {MODELS} | seeds {N_SEEDS}', '| SMOKE' if SMOKE else '', flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X_flat, Y = load_breakthrough_states(SOURCE_STATES)            # position-only features, (N, 2 * side^2)
    n_all = len(X_flat)
    counts = read_counts(FORGETTING, N_SMOKE if SMOKE else n_all)
    if SMOKE:
        X_flat, Y = X_flat[:N_SMOKE], Y[:N_SMOKE]
    assert len(counts) == len(X_flat), f'{FORGETTING}: {len(counts)} rows, E has {len(X_flat)}'
    board_size = board_size_of(X_flat)
    print(f'loaded {len(X_flat)} states, forgetting count 0: {(counts == 0).mean():.1%}, max {counts.max()}', flush=True)

    X_t = torch.as_tensor(X_flat, dtype=torch.float32, device=device)
    Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)
    order = get_order(ORDERS_DIR / f'{METHOD}_position_only.txt', X_flat, METHOD, RATIO, device)
    idx = torch.as_tensor(order[:int(RATIO * len(X_t))], device=device)
    in_s = np.zeros(len(X_t), bool)
    in_s[idx.cpu().numpy()] = True
    X_S, Y_S = X_t[idx], Y_t[idx]

    summary, bins, gains, cal_summary, cal_gains = [], [], [], [], []
    rng = np.random.default_rng(SEED)
    pool_idx = torch.as_tensor(np.flatnonzero(~in_s), device=device)
    for kind in MODELS:
        for k in range(N_SEEDS):
            seed = SEED + k
            torch.manual_seed(seed)
            model = build_model(kind, X_S, on_code=False, board_size=board_size).to(device)
            train(model, X_S, Y_S)
            err = sq_errors(model, X_t, Y_t)
            train_mse = mse(model, X_S, Y_S, EVAL_BS)
            spearman = pd.Series(counts).corr(pd.Series(err), method='spearman')
            summary.append(dict(model=kind, seed=seed, mse_E=err.mean(), train_mse=train_mse, spearman=spearman,
                                top1_error_share=share_of_top(err, counts, 0.01), top5_error_share=share_of_top(err, counts, 0.05),
                                oracle_top5_error_share=np.sort(err)[::-1][:int(0.05 * len(err))].sum() / err.sum()))
            bins += bin_rows(err, counts, in_s, kind, seed)
            gains += gain_rows(err, counts, kind, seed)
            print(summary[-1], flush=True)
            if CAL:
                scores = {space: CAL_scores(model, X_S, X_t[pool_idx], CAL_K, bs=EVAL_BS, space=space).cpu().numpy().astype(np.float64)
                          for space in CAL_SPACES}
                rows, curves = cal_rows(err, counts, in_s, scores, kind, seed, rng)
                cal_summary += rows
                cal_gains += curves
                print(pd.DataFrame(rows)[['sorter', 'spearman', 'top1_error_share', 'top5_error_share', 'top5_precision_hard']].round(3).to_string(index=False), flush=True)
            del model
            torch.cuda.empty_cache()

    pd.DataFrame(summary).to_csv(OUT_DIR / 'summary.csv', index=False)
    pd.DataFrame(bins).to_csv(OUT_DIR / 'bins.csv', index=False)
    pd.DataFrame(gains).to_csv(OUT_DIR / 'gain_curve.csv', index=False)
    if CAL:
        pd.DataFrame(cal_summary).to_csv(OUT_DIR / 'cal_summary.csv', index=False)
        pd.DataFrame(cal_gains).to_csv(OUT_DIR / 'cal_gain_curve.csv', index=False)
        print(pd.DataFrame(cal_summary).groupby(['model', 'sorter'], sort=False)[['spearman', 'top1_error_share', 'top5_error_share',
              'top5_precision_hard', 'top5_recall_hard']].mean().round(3))
    print(pd.DataFrame(bins).query("subset == 'all'").groupby(['model', 'bin'], sort=False)[['state_share', 'mean_error', 'error_share']].mean().round(4))
    print(f'wrote {OUT_DIR}')


if __name__ == '__main__':
    main()
