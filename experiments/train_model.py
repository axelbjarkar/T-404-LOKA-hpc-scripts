"""Train ONE CNN on the first 10% of o.txt as ordered by herding on the near-goal embedding (MSE, Adam)
and save its weights as a .pt file.

Reuses herding_with_near_goal_10p.txt written by order_embeddings_10p.py (computes it if missing).
As in train_mse_all.py, the model is fed the same embedding the subset was selected with
(positions + [white near goal, black near goal] -> CNN with 2 extra constant planes)."""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from loka.data import load_breakthrough_states, save_breakthrough_states
from loka.embeddings import with_near_goal
from loka.geometrical_methods import herding_order
from loka.models import get_breakthrough_cnn

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data" / "game_states" / f"{os.environ.get('LOKA_SOURCE', 'o')}.txt"
EMB_DIR       = ROOT / "data/ordered-data/embeddings"   # shared with order_embeddings_10p.py
SUBSET_SIZE   = 0.10                                    # subset size as a share of E
ORDERED_FILE  = EMB_DIR / f"herding_{with_near_goal.__name__}_{round(SUBSET_SIZE * 100)}p.txt"
MODEL_FILE    = EMB_DIR / "models" / f"{ORDERED_FILE.stem}.pt"

# ---- CONFIG: experiment settings ----
EPOCHS = 50
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


def main():
    print(device)
    if MODEL_FILE.exists():
        print(f'skipping: {MODEL_FILE} already exists')
        return
    os.makedirs(MODEL_FILE.parent, exist_ok=True)

    X, Y = load_breakthrough_states(SOURCE_STATES, two_channel=True)
    print(f'loaded {len(X)} states from {SOURCE_STATES}')
    S_size = int(SUBSET_SIZE * len(X))

    if ORDERED_FILE.exists():
        X_S, Y_S = load_breakthrough_states(ORDERED_FILE)
        assert len(X_S) >= S_size, f'{ORDERED_FILE} is shorter than SUBSET_SIZE * |E|'
        print(f'loaded herding (near-goal) subset from {ORDERED_FILE}')
    else:
        order = herding_order(with_near_goal(X), SUBSET_SIZE, device)
        X_S, Y_S = X[order], Y[order]
        save_breakthrough_states(ORDERED_FILE, X_S, Y_S)
        print(f'wrote {ORDERED_FILE}')

    # train on the near-goal embedding of the first S_size herding picks
    X_S = torch.FloatTensor(with_near_goal(X_S[:S_size])).to(device)
    Y_S = torch.FloatTensor(Y_S[:S_size]).to(device)
    print(f'training on {len(X_S)} states, input dim {X_S.shape[1]}')

    torch.manual_seed(SEED)
    model = get_breakthrough_cnn(X_S).to(device)  # 2 board channels + 2 near-goal planes
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

    # evaluate on all of E through the same embedding
    X_t = torch.FloatTensor(with_near_goal(X)).to(device)
    Y_t = torch.FloatTensor(Y).to(device)
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
