"""Experiments that decide what to build next. All scored on held-out DFL games, never BAR-ATH.

    python -m pipeline.experiments ball_upper_bound
    python -m pipeline.experiments self_training

ball_upper_bound: how much spellF1 is lost to the missing ball? Train and evaluate with the ball
    never dropped (all other noise kept). The gap to the realistic setting bounds what a video
    ball detector could recover.
self_training: does pseudo-labelling the target game help? For each held-out DFL game, take the
    decoded out-of-fold labels as pseudo-labels on its frames, refit with them, re-score.
"""

import json
import os
import sys
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

from . import data, decode, features, noise, train


def _noisy_features(args):
    match_id, realisation, drop_ball = args
    stats, clean = train._noise_stats()
    if not drop_ball:
        stats = dict(stats, p_lose=[0.0] * len(stats["p_lose"]))
    rng = np.random.default_rng([train.SEED, data.DFL_MATCHES.index(match_id), realisation, 7])
    m = data.load_dfl(match_id)
    noisy = noise.apply(m, stats, clean, rng)
    _, xa = features.compute(noisy)
    _, xb = features.compute(noisy, swap=True)
    y = decode.controlled_labels(m.labels, m.segments, train.MIN_SPELL)
    return match_id, realisation, xa, xb, y, m.segments


def ball_upper_bound():
    jobs = [(m, r, False) for m in data.DFL_MATCHES for r in range(2)]
    with Pool(7) as pool:
        out = pool.map(_noisy_features, jobs)
    games = {(m, r): (xa, xb, y, segs) for m, r, xa, xb, y, segs in out}
    probs = {}
    for held in data.DFL_MATCHES:
        xs, ys = [], []
        for m in data.DFL_MATCHES:
            if m == held:
                continue
            xa, xb, y, _ = games[(m, 1)]
            idx = np.flatnonzero(y != data.X)[::train.STRIDE]
            xs += [xa[idx], xb[idx]]
            ys += [y[idx], features.swap_labels(y)[idx]]
        model = lgb.train(train.LGB_PARAMS, lgb.Dataset(np.concatenate(xs), np.concatenate(ys)),
                          num_boost_round=train.N_ROUNDS)
        xa, xb, _, _ = games[(held, 0)]
        probs[held] = train.predict_proba(model, xa, xb)
        print("  fold", held, flush=True)
    with open(os.path.join(train.MODEL_DIR, "decoder.json")) as handle:
        params = json.load(handle)
    # Score against the same targets the realistic run uses.
    s, mf, per = train.score(data.DFL_MATCHES, probs, params)
    print("ball never missing: spellF1 {:.4f} macroF1 {:.4f} (decoder from realistic run)".format(
        s, mf))


def gru_cv():
    """Leave-one-game-out GRU, then compare LightGBM, GRU and their average on held-out games."""
    import time
    from . import seq
    oof = {}
    for held in data.DFL_MATCHES:
        t = time.time()
        games = []
        for m in data.DFL_MATCHES:
            if m == held:
                continue
            for r in range(1, train.N_TRAIN_NOISE + 1):
                g = train._load(m, r)
                games.append((g["xa"], g["xb"], g["y"], [tuple(s) for s in g["segments"]]))
        model = seq.fit(games)
        g = train._load(held, 0)
        oof[held] = seq.predict(model, g["xa"], g["xb"], [tuple(s) for s in g["segments"]])
        print("  gru fold {} {:.0f}s".format(held, time.time() - t), flush=True)
    gru_path = os.path.join(train.MODEL_DIR, "oof_gru.npz")
    np.savez(gru_path, **oof)
    lgb_oof = np.load(os.path.join(train.MODEL_DIR, "oof_probs.npz"))
    avg_path = os.path.join(train.MODEL_DIR, "oof_avg.npz")
    np.savez(avg_path, **{m: (lgb_oof[m] + oof[m]) / 2 for m in data.DFL_MATCHES})
    for name, path in (("lightgbm", None), ("gru", gru_path), ("average", avg_path)):
        s, mf, params = train.tune_decoder(path)
        print("{:<9} spellF1 {:.4f} macroF1 {:.4f} {}".format(name, s, mf, params), flush=True)


PSEUDO_CONF = 0.9


def pseudo_rows(xa, xb, probs, decoded):
    """Confident frames of the target game, labelled by the model's own decoded output."""
    conf = (probs.max(axis=1) > PSEUDO_CONF) & (probs.argmax(axis=1) == decoded) \
        & (decoded != data.X)
    idx = np.flatnonzero(conf)[::train.STRIDE]
    return (np.concatenate([xa[idx], xb[idx]]),
            np.concatenate([decoded[idx], features.swap_labels(decoded)[idx]]))


def self_training():
    """Simulate the self-training procedure on each held-out DFL game."""
    import time
    with open(os.path.join(train.MODEL_DIR, "decoder.json")) as handle:
        params = json.load(handle)
    oof = dict(np.load(os.path.join(train.MODEL_DIR, "oof_probs.npz")))
    new = {}
    for held in data.DFL_MATCHES:
        t = time.time()
        g = train._load(held, 0)
        segs = [tuple(s) for s in g["segments"]]
        decoded = decode.decode(oof[held], segs, params)
        px, py = pseudo_rows(g["xa"], g["xb"], oof[held], decoded)
        xs, ys = [px], [py]
        for m in data.DFL_MATCHES:
            if m == held:
                continue
            for r in range(1, train.N_TRAIN_NOISE + 1):
                x, y = train._rows(train._load(m, r))
                xs.append(x)
                ys.append(y)
        model = lgb.train(train.LGB_PARAMS, lgb.Dataset(np.concatenate(xs), np.concatenate(ys)),
                          num_boost_round=train.N_ROUNDS)
        new[held] = train.predict_proba(model, g["xa"], g["xb"])
        print("  self-train fold {} ({} pseudo rows) {:.0f}s".format(held, len(py),
                                                                     time.time() - t), flush=True)
    path = os.path.join(train.MODEL_DIR, "oof_selftrain.npz")
    np.savez(path, **new)
    base = train.score(data.DFL_MATCHES, oof, params)
    same = train.score(data.DFL_MATCHES, new, params)
    print("baseline          spellF1 {:.4f} macroF1 {:.4f}".format(base[0], base[1]))
    print("self-trained      spellF1 {:.4f} macroF1 {:.4f} (same decoder)".format(same[0], same[1]))
    s, mf, p = train.tune_decoder(path)
    print("self-trained      spellF1 {:.4f} macroF1 {:.4f} (re-tuned decoder {})".format(s, mf, p))


if __name__ == "__main__":
    {"ball_upper_bound": ball_upper_bound, "gru_cv": gru_cv,
     "self_training": self_training}[sys.argv[1]]()
