"""Shared helpers for the HPC experiments (import with `from common import ...`; scripts run from experiments/).

- mse(): batched MSE, used for both the training-set MSE and the MSE on all of E
- FORCE / fresh(): LOKA_FORCE=1 recomputes cached files (orders, autoencoder codes, results) once per process
- EMBEDDINGS: name -> function(flat (N, 50) positions) -> (N, 50 + k) features
- get_order(): herding / k-center order, computed once and cached on disk
- build_autoencoder() / get_codes(): conv autoencoder (as in dev/M2/AUTO ENCODER.ipynb), 10-d bottleneck codes cached on disk
- build_model(): CNN / Residual for board features (2D), CNN1D / Residual1D for the 10-d codes
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loka.embeddings import counts, near_goal, clear_lane, with_counts, with_near_goal, with_clear_lane
from loka.geometrical_methods import herding_order, k_center_order
from loka.models import (AutoEncoder, BreakthroughModel, BreakthroughResNet, N_BOARD_FEATURES,
                         get_breakthrough_cnn)

FORCE = os.environ.get("LOKA_FORCE", "0") not in ("", "0")
_recomputed = set()   # paths already recomputed in this process (so a forced file is not redone per experiment)


def fresh(path):
    """True if `path` must be (re)computed: it is missing, or LOKA_FORCE is set and it was not recomputed yet."""
    path = Path(path)
    if not path.exists():
        _recomputed.add(path)
        return True
    if FORCE and path not in _recomputed:
        _recomputed.add(path)
        return True
    return False


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def mse(model, X, Y, bs=1024):
    """Mean squared error of `model` on (X, Y), batched, in eval mode (restores the previous mode)."""
    was_training = model.training
    model.eval()
    sq_err = 0.0
    for i in range(0, len(X), bs):
        sq_err += F.mse_loss(model(X[i:i + bs]), Y[i:i + bs], reduction='sum').item()
    model.train(was_training)
    return sq_err / Y.numel()


# ---------------------------------------------------------------- embeddings
def _positions(X):
    return X.reshape(len(X), -1)


def _all_three(X):
    return np.concatenate([_positions(X), counts(X), near_goal(X), clear_lane(X)], axis=1)   # (N, 56)


EMBEDDINGS = {   # name -> function(X) where X is (N, 50) or (N, 2, 5, 5)
    'position_only': _positions,
    'piece_count':   with_counts,
    'near_goal':     with_near_goal,
    'clear_lane':    with_clear_lane,
    'all_three':     _all_three,
}


# ---------------------------------------------------------------- orders
def get_order(path, X_feat, method, ratio, device, rep=None):
    """Order (indexes into X_feat) for 'herding' or 'k_center'; loaded from `path` unless missing / forced.
    Only the prefix of length ratio * |X| is computed."""
    path = Path(path)
    if not fresh(path):
        order = np.loadtxt(path, dtype=np.int64)
        assert len(order) >= int(ratio * len(X_feat)), f'{path} is shorter than ratio * |E|'
        print(f'loaded {method} order from {path}', flush=True)
        if rep is not None:
            rep.add_data(f'order {path.stem}', path)
        return order
    fn = {'herding': herding_order, 'k_center': k_center_order}[method]
    order = fn(X_feat, ratio, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, order, fmt='%d')
    print(f'wrote {path}', flush=True)
    if rep is not None:
        rep.add_output(path, method=method)
    return order


# ---------------------------------------------------------------- autoencoder
class ToPlanes(nn.Module):
    """(N, 50 + k) -> (N, 2 + k, 5, 5): board as 2 channels, each extra field as a constant plane."""
    def forward(self, x):
        board = x[:, :N_BOARD_FEATURES].reshape(-1, 2, 5, 5)
        extra = x[:, N_BOARD_FEATURES:, None, None].expand(-1, -1, 5, 5)
        return torch.cat([board, extra], dim=1)


class FromPlanes(nn.Module):
    """(N, 2 + k, 5, 5) -> (N, 50 + k): board channels flattened, each extra plane averaged to one value."""
    def forward(self, x):
        return torch.cat([x[:, :2].flatten(1), x[:, 2:].mean(dim=(2, 3))], dim=1)


def build_autoencoder(n_features, bottleneck=10):
    """Conv autoencoder from AUTO ENCODER.ipynb for (N, n_features) inputs; works for 50 (positions) and 50 + k."""
    C = 2 + (n_features - N_BOARD_FEATURES)
    encoder = nn.Sequential(
        ToPlanes(),
        nn.Conv2d(C, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, bottleneck), nn.ReLU(),
    )
    decoder = nn.Sequential(
        nn.Linear(bottleneck, 32 * 5 * 5), nn.ReLU(),
        nn.Unflatten(1, (32, 5, 5)),
        nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(16, C, kernel_size=3, padding=1),
        FromPlanes(),
    )
    return AutoEncoder(encoder, decoder)


def get_codes(path, X_feat, device, epochs=50, lr=1e-3, seed=42, bottleneck=10, rep=None, desc=''):
    """(N, bottleneck) float32 codes of the autoencoder trained on all of X_feat; cached as .npy at `path`."""
    from tqdm import tqdm
    path = Path(path)
    if not fresh(path):
        print(f'loaded codes from {path}', flush=True)
        if rep is not None:
            rep.add_data(f'codes {path.stem}', path)
        return np.load(path)
    X_t = torch.as_tensor(X_feat, dtype=torch.float32, device=device)
    torch.manual_seed(seed)
    ae = build_autoencoder(X_t.shape[1], bottleneck).to(device)
    if rep is not None:
        rep.add_model(f'autoencoder_{path.stem}', ae, input_shape=[X_t.shape[1]])
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    with tqdm(total=epochs, desc=f'autoencoder {desc}') as pbar:
        ae.train_model(X_t, optimizer, nn.MSELoss(), epochs=epochs, verbose=True, pbar=pbar)
    ae.eval()
    with torch.no_grad():
        codes = torch.cat([ae.extract_encoder()(X_t[i:i + 1024]) for i in range(0, len(X_t), 1024)])
        recon = F.mse_loss(ae(X_t[:1024]), X_t[:1024]).item()
    print(f'autoencoder reconstruction MSE (first 1024 states): {recon:.6f}', flush=True)
    codes = codes.cpu().numpy().astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, codes)
    print(f'wrote {path}', flush=True)
    if rep is not None:
        rep.add_output(path, reconstruction_mse=recon)
    del X_t, ae
    torch.cuda.empty_cache()
    return codes


# ---------------------------------------------------------------- models on the 10-d code
class BreakthroughCNN1D(BreakthroughModel):
    """BreakthroughCNN with Conv1d over the code: (N, L) -> (N, 1, L). Same widths and head as the 2D CNN."""
    def __init__(self, length=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * length, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def preprocess(self, x):
        return x.reshape(len(x), 1, -1)


class ResBlock1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        y = torch.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return torch.relu(x + y)


class BreakthroughResNet1D(BreakthroughModel):
    """BreakthroughResNet with Conv1d / BatchNorm1d over the code. Same channels, blocks and head sizes."""
    def __init__(self, length=10, channels=32, blocks=2, head_ch=64, hidden=115):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(),
            *[ResBlock1D(channels) for _ in range(blocks)],
            nn.Conv1d(channels, head_ch, 1, bias=False),
            nn.BatchNorm1d(head_ch), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_ch * length, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def preprocess(self, x):
        return x.reshape(len(x), 1, -1)


def build_model(kind, X, on_code):
    """kind: 'cnn' or 'residual'. X: (N, F) training features. on_code: X is an autoencoder code (1D models)
    instead of 50 + k board features (2D models, input channels sized to X)."""
    F_ = X.reshape(len(X), -1).shape[1]
    if on_code:
        return {'cnn': BreakthroughCNN1D, 'residual': BreakthroughResNet1D}[kind](length=F_)
    if kind == 'cnn':
        return get_breakthrough_cnn(X)
    return BreakthroughResNet(n_extra=F_ - N_BOARD_FEATURES)
