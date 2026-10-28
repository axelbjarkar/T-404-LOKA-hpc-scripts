"""Order o.txt with herding / k-center on positions only (ported from 'ordering large data.ipynb')."""
import os
from pathlib import Path

import torch

from loka.data import load_breakthrough_states, save_breakthrough_states
from loka.geometrical_methods import herding_order, k_center_order

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/o.txt"
ORDERED_DIR   = ROOT / "data/ordered-data"

# ---- CONFIG: experiment settings ----
RATIO = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ORDERS = [
    ('herding_25p.txt',          lambda D: herding_order(D, RATIO, device)),
    ('k_center_no_S0_25p.txt',   lambda D: k_center_order(D, RATIO, device, S0=None)),
    ('k_center_rand_S0_25p.txt', lambda D: k_center_order(D, RATIO, device, S0='rand')),
]


def main():
    print(device)
    os.makedirs(ORDERED_DIR, exist_ok=True)

    X, Y = load_breakthrough_states(SOURCE_STATES, two_channel=True)
    print(f'loaded {len(X)} states from {SOURCE_STATES}')

    for fname, method in ORDERS:
        path = ORDERED_DIR / fname
        if path.exists():
            print(f'skipping {fname}: {path} already exists')
            continue

        order = method(X)
        save_breakthrough_states(path, X[order], Y[order])
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
