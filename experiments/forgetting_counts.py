"""Count forgetting events for every state of E: train the CNN on all of E and record, per state,
how often it goes from correctly to incorrectly classified between epochs
(ported from 'Counting Forgetting events.ipynb')."""
import os
from pathlib import Path

import torch
from tqdm import tqdm

from loka.data import load_breakthrough_states, save_forgetting_states
from loka.report import ExperimentReport, config_from_globals
from loka.models import get_breakthrough_cnn
from loka.training_methods import ForgettingScores

# ---- CONFIG: input / output locations ----
ROOT          = Path(os.environ.get("LOKA_ROOT", "/CHANGE/ME/T-404-LOKA"))
SOURCE_STATES = ROOT / "data/game_states/o.txt"                 # all of E
OUT_FILE      = ROOT / "data/forgetting_counts/forgetting_o.txt"

# ---- CONFIG: experiment settings ----
EPOCHS = 200
BATCH_SIZE = 512
LR = 0.001
SEED = 42
OPTIMIZER = "Adam"   # torch.optim name

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    print(device)
    os.makedirs(OUT_FILE.parent, exist_ok=True)

    with ExperimentReport('forgetting_counts', config=config_from_globals(globals())) as rep:
        if OUT_FILE.exists():
            print(f'skipping: {OUT_FILE} already exists')
            rep.add_skipped(OUT_FILE)
            return

        rep.add_data('source_states (E)', SOURCE_STATES)
        X, Y = load_breakthrough_states(SOURCE_STATES)
        print(f'loaded {len(X)} states from {SOURCE_STATES}')

        model = get_breakthrough_cnn(X)
        rep.add_model('forgetting', model, input_shape=list(X.shape[1:]), embedding='positions_only')
        forgetter = ForgettingScores(model, seed=SEED)

        pbar = tqdm(total=EPOCHS, desc='forgetting counts')
        forgetter.train(X=X, Y=Y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                        optimizer=OPTIMIZER, device=device, pbar=pbar)
        pbar.close()

        C = forgetter.getForgettingCounts().cpu().numpy()
        assert len(C) == len(X), f'got {len(C)} counts for {len(X)} states'
        print(f'forgetting counts: min={C.min()} max={C.max()} mean={C.mean():.3f} '
              f'never forgotten={(C == 0).mean():.1%}')
        rep.log_metric('forgetting/mean_count', float(C.mean()))
        rep.log_metric('forgetting/max_count', float(C.max()))

        save_forgetting_states(OUT_FILE, X, Y, C)
        print(f'wrote {OUT_FILE}')
        rep.add_output(OUT_FILE, input='source_states (E)')


if __name__ == '__main__':
    main()
