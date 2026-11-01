"""Order e.txt with herding on positions only, for the M2 experiments on E = e.txt
(same method as order_positions_25p.py, which orders o.txt)."""
import os
from pathlib import Path

import torch

from loka.data import load_breakthrough_states, save_breakthrough_states
from loka.geometrical_methods import herding_order
from loka.report import ExperimentReport, config_from_globals

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/e.txt"
OUT_FILE      = ROOT / "data/ordered-data/herding_e_25p.txt"

# ---- CONFIG: experiment settings ----
RATIO = 0.25

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(device)
    os.makedirs(OUT_FILE.parent, exist_ok=True)

    with ExperimentReport('order_herding_e_25p', config=config_from_globals(globals())) as rep:
        if OUT_FILE.exists():
            print(f'skipping: {OUT_FILE} already exists')
            rep.add_skipped(OUT_FILE)
            return

        rep.add_data('source_states', SOURCE_STATES)
        X, Y = load_breakthrough_states(SOURCE_STATES, two_channel=True)
        print(f'loaded {len(X)} states from {SOURCE_STATES}')

        order = herding_order(X, RATIO, device)
        save_breakthrough_states(OUT_FILE, X[order], Y[order])
        print(f'wrote {OUT_FILE}')
        rep.add_output(OUT_FILE, ratio=RATIO)


if __name__ == '__main__':
    main()
