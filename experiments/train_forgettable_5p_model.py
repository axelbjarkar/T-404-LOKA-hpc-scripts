"""Train ONE CNN as in partial_forgettable.py's forgettable_5p run (herding prefix of E where the last 5% of the
slots are forgettable events, MSE, Adam) and save its weights as a .pt file.

Reuses forgetting_counts.txt and herding_order.txt written by partial_forgettable.py (computes them if missing).
Subset size (share of E) and ratio are set by SUBSET_SIZE / FORGETTABLE_RATIO below."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states, load_forgetting_states, save_forgetting_states
from loka.geometrical_methods import herding_order
from loka.models import get_breakthrough_cnn
from loka.training_methods import ForgettingScores, make_forgettable_subset

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data" / "game_states" / f"{os.environ.get('LOKA_SOURCE', 'o')}.txt"
EXPERIMENT_DIR = ROOT / "data/partial_forgettable" / SOURCE_STATES.stem   # shared with partial_forgettable.py
COUNTS_FILE   = EXPERIMENT_DIR / "forgetting_counts.txt"
ORDER_FILE    = EXPERIMENT_DIR / "herding_order.txt"
MODEL_FILE    = EXPERIMENT_DIR / "models" / "forgettable_5p.pt"

# ---- CONFIG: experiment settings ----
FORGETTABLE_RATIO = 0.05   # share of the subset (from the end) made forgettable
SUBSET_SIZE = 0.1          # subset size as a share of E (END of the partial_forgettable sweep)
EPOCHS = 50
FORGETTING_EPOCHS = 200
BATCH_SIZE = 512
EVAL_BS = 1024
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"
LOSS = "MSELoss"

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def forgetting_counts(X, Y):
    model = get_breakthrough_cnn(X)
    forgetter = ForgettingScores(model, seed=SEED)
    pbar = tqdm(total=FORGETTING_EPOCHS, desc='forgetting counts')
    forgetter.train(X=X, Y=Y, epochs=FORGETTING_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                    optimizer=OPTIMIZER, device=device, pbar=pbar)
    pbar.close()
    return forgetter.getForgettingCounts()


def main():
    print(device)
    if MODEL_FILE.exists():
        print(f'skipping: {MODEL_FILE} already exists')
        return
    os.makedirs(MODEL_FILE.parent, exist_ok=True)

    X, Y = load_breakthrough_states(SOURCE_STATES)
    print(f'loaded {len(X)} states from {SOURCE_STATES}')

    if COUNTS_FILE.exists():
        X_c, Y_c, C = load_forgetting_states(COUNTS_FILE)
        assert np.array_equal(X_c, X) and np.array_equal(Y_c, Y), f'{COUNTS_FILE} does not match {SOURCE_STATES}'
        C = torch.as_tensor(C, dtype=torch.int32, device=device)
        print(f'loaded forgetting counts from {COUNTS_FILE}')
    else:
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)
        C = forgetting_counts(X, Y)
        save_forgetting_states(COUNTS_FILE, X, Y, C.cpu().numpy())
        print(f'wrote {COUNTS_FILE}')

    if ORDER_FILE.exists():
        order = np.loadtxt(ORDER_FILE, dtype=np.int64)
        assert len(order) >= int(SUBSET_SIZE * len(X)), f'{ORDER_FILE} is shorter than SUBSET_SIZE * |E|'
        print(f'loaded herding order from {ORDER_FILE}')
    else:
        order = herding_order(X, SUBSET_SIZE, device)
        np.savetxt(ORDER_FILE, order, fmt='%d')
        print(f'wrote {ORDER_FILE}')

    X_t = torch.FloatTensor(X).to(device)
    Y_t = torch.FloatTensor(Y).to(device)

    S_size = int(SUBSET_SIZE * len(X_t))
    forgettable_size = int(S_size * FORGETTABLE_RATIO)
    X_S, Y_S = make_forgettable_subset(X_t, Y_t, C, order, S_size, forgettable_size)
    print(f'training on {len(X_S)} states ({forgettable_size} forgettable)')

    torch.manual_seed(SEED)
    model = get_breakthrough_cnn(X_S).to(device)
    optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), lr=LR)
    loss_fn = getattr(nn, LOSS)()

    for epoch in tqdm(range(EPOCHS), desc='training'):
        perm = torch.randperm(len(X_S), device=device)
        for i in range(0, len(X_S), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(X_S[idx]), Y_S[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    sq_err = 0.0
    with torch.no_grad():
        for i in range(0, len(X_t), EVAL_BS):
            sq_err += F.mse_loss(model(X_t[i:i + EVAL_BS]), Y_t[i:i + EVAL_BS], reduction='sum').item()
    print(f'final batch loss {loss.item():.4f}, MSE on E {sq_err / Y_t.numel():.6f}')

    torch.save(model.state_dict(), MODEL_FILE)
    print(f'wrote {MODEL_FILE}')


if __name__ == '__main__':
    main()
