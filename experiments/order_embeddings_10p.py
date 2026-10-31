"""Order o.txt with herding / k-center on the feature embeddings (ported from 'Ordering_after_different embeddings.ipynb')."""
import os
from pathlib import Path

import torch

from loka.data import load_breakthrough_states, save_breakthrough_states
from loka.geometrical_methods import herding_order, k_center_order
from loka.embeddings import with_counts, with_near_goal, with_counts_and_near_goal
from loka.report import ExperimentReport, config_from_globals

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/o.txt"
EMB_DIR       = ROOT / "data/ordered-data/embeddings"

# ---- CONFIG: experiment settings ----
RATIO = 0.10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SUBSET_METHODS = {
    'herding':          lambda E: herding_order(E, RATIO, device),
    'k_center_no_S0':   lambda E: k_center_order(E, RATIO, device, S0=None),
    'k_center_rand_S0': lambda E: k_center_order(E, RATIO, device, S0='rand'),
}
EMBEDDING_METHODS = [
    with_counts,
    with_near_goal,
    with_counts_and_near_goal,
]


def main():
    print(device)
    os.makedirs(EMB_DIR, exist_ok=True)

    with ExperimentReport('order_embeddings_10p', config=config_from_globals(globals())) as rep:
        rep.add_data('source_states', SOURCE_STATES)
        X, Y = load_breakthrough_states(SOURCE_STATES, two_channel=True)
        print(f'loaded {len(X)} states from {SOURCE_STATES}')

        for subset_name, subset_fn in SUBSET_METHODS.items():
            for emb in EMBEDDING_METHODS:
                fname = f"{subset_name}_{emb.__name__}_{round(RATIO * 100)}p.txt"
                path = EMB_DIR / fname
                if path.exists():
                    print(f'skipping {fname}: {path} already exists')
                    rep.add_skipped(path)
                    continue

                order = subset_fn(emb(X))
                save_breakthrough_states(path, X[order], Y[order])  # saves positions; train re-applies emb
                print(f'wrote {path}')
                rep.add_output(path, method=subset_name, embedding=emb.__name__, ratio=RATIO)


if __name__ == '__main__':
    main()
