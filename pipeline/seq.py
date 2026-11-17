"""A bidirectional GRU over the per-frame features: long-range context for the same labels.

LightGBM sees each frame through hand-made windows of up to +/-5 s. A recurrent model can carry
context for as long as it is useful -- which side has been building an attack for twenty seconds,
whether a stoppage has already been followed by a restart. Its probabilities are averaged with
LightGBM's before decoding; whether that helps is decided by leave-one-game-out CV on DFL.

Runs on the CPU with deterministic algorithms, so the submission regenerates exactly.
"""

import os

import numpy as np
import torch
from torch import nn

from . import data, features

STEP = 5                 # frames: the GRU runs at 5 fps
CHUNK = 256              # steps per training sequence (~51 s)
EPOCHS = 6
BATCH = 32
HIDDEN = 64
SEED = 2027


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.1)
        self.out = nn.Linear(2 * HIDDEN, 3)

    def forward(self, x):
        h, _ = self.gru(self.inp(x))
        return self.out(h)


def _setup(seed=SEED):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(12)


class Scaler:
    def __init__(self, x):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0) + 1e-6

    def __call__(self, x):
        z = (x - self.mean) / self.std
        return np.nan_to_num(np.clip(z, -6, 6), nan=0.0).astype(np.float32)


def _sequences(x, y, segments):
    """Per segment, the 5-fps sequence of features and labels."""
    out = []
    for start, stop in segments:
        idx = np.arange(start, stop, STEP)
        out.append((x[idx], y[idx] if y is not None else None))
    return out


def fit(games, seed=SEED):
    """games: list of (xa, xb, y, segments) training realisations."""
    _setup(seed)
    rng = np.random.default_rng(seed)
    seqs = []
    for xa, xb, y, segs in games:
        seqs += _sequences(xa, y, segs)
        seqs += _sequences(xb, features.swap_labels(y), segs)
    scaler = Scaler(np.concatenate([s[0] for s in seqs])[::7])
    seqs = [(scaler(x), y) for x, y in seqs]
    starts = [(i, s) for i, (x, _) in enumerate(seqs)
              for s in range(0, max(1, len(x) - CHUNK), CHUNK)]
    net = Net(seqs[0][0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-3, total_steps=EPOCHS * ((len(starts) + BATCH - 1) // BATCH))
    loss_fn = nn.CrossEntropyLoss(ignore_index=data.X)
    net.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(starts))
        for b in range(0, len(order), BATCH):
            xs, ys = [], []
            for k in order[b:b + BATCH]:
                i, s = starts[k]
                x, y = seqs[i]
                xs.append(x[s:s + CHUNK])
                ys.append(y[s:s + CHUNK].astype(np.int64))
            n = min(len(v) for v in xs)
            xb = torch.from_numpy(np.stack([v[:n] for v in xs]))
            yb = torch.from_numpy(np.stack([v[:n] for v in ys]))
            opt.zero_grad()
            loss = loss_fn(net(xb).reshape(-1, 3), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
    net.eval()
    return net, scaler


def _predict_one(net, scaler, x, segments, n_frames):
    probs = np.full((n_frames, 3), np.nan)
    with torch.no_grad():
        for start, stop in segments:
            idx = np.arange(start, stop, STEP)
            z = torch.from_numpy(scaler(x[idx]))[None]
            p = torch.softmax(net(z)[0], dim=-1).numpy()
            t = np.arange(start, stop)
            for k in range(3):
                probs[start:stop, k] = np.interp(t, idx, p[:, k])
    return probs


def predict(model, xa, xb, segments):
    net, scaler = model
    n = len(xa)
    pa = _predict_one(net, scaler, xa, segments, n)
    pb = _predict_one(net, scaler, xb, segments, n)[:, [1, 0, 2]]
    return (pa + pb) / 2.0


def save(model, path):
    net, scaler = model
    torch.save({"state": net.state_dict(), "mean": scaler.mean, "std": scaler.std,
                "n_in": len(scaler.mean)}, path)


def load(path):
    blob = torch.load(path, weights_only=False)
    net = Net(blob["n_in"])
    net.load_state_dict(blob["state"])
    net.eval()
    scaler = Scaler.__new__(Scaler)
    scaler.mean, scaler.std = blob["mean"], blob["std"]
    return net, scaler
