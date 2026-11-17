"""Train the possession model on the open DFL games and choose the decoder on held-out DFL games.

    python -m pipeline.train            # writes models/

Steps:
  1. Measure the noise statistics of the provided tracking (aggregates only, no labels).
  2. Degrade every DFL game to match them, compute features for both team orientations.
  3. Leave-one-game-out cross-validation -> out-of-fold probabilities for all seven games.
  4. Grid-search the Viterbi costs on those out-of-fold probabilities, maximising spellF1 with
     the official scorer. This is the only tuning, and it never touches the provided game.
  5. Fit the final model on all seven games.

Deterministic: fixed seeds, LightGBM in deterministic mode, so the same models come out.
"""

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np

from . import data, decode, features, noise

ROOT = data.ROOT
sys.path.insert(0, os.path.join(data.CHALLENGE, "evaluation"))
import possession_metrics as pm  # noqa: E402

# MTAH_TAGS=1 adds the Smart-Tagging restart inputs (pipeline/tags.py): simulated tags for the DFL
# games, the real Smart Tagging for the provided game. Its models go to their own folder.
USE_TAGS = os.environ.get("MTAH_TAGS", "0") == "1"
BASE_MODEL_DIR = os.path.join(ROOT, "models", "possession")
MODEL_DIR = os.path.join(ROOT, "models", "possession_tags") if USE_TAGS else BASE_MODEL_DIR
TAG_STATS = os.path.join(BASE_MODEL_DIR, "tag_stats.json")
FEAT_DIR = os.path.join(data.CACHE_DIR, "features")
SEED = 2027
STRIDE = 3               # training rows: every 3rd frame (neighbours are near-duplicates)
N_TRAIN_NOISE = 2        # noise realisations per game for training
LGB_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, deterministic=True,
                  force_row_wise=True, num_threads=20)
N_ROUNDS = 400
MIN_SPELL = 25           # frames: CHALLENGE.md "controlled possession", see decode.py


def _noise_stats():
    path = os.path.join(MODEL_DIR, "noise_stats.json")
    clean_path = os.path.join(MODEL_DIR, "noise_stats_dfl.json")
    if not os.path.exists(path):
        noise.save_stats(noise.measure(data.load_atd()), path)
    if not os.path.exists(clean_path):
        per_game = [noise.measure(data.load_dfl(m)) for m in data.DFL_MATCHES]
        clean = {k: float(np.mean([g[k] for g in per_game]))
                 for k in ("player_jitter", "ball_jitter")}
        noise.save_stats(clean, clean_path)
    return noise.load_stats(path), noise.load_stats(clean_path)


def _game_features(args):
    """Features for one DFL game: realisation r of the noise, both orientations."""
    match_id, realisation, stats, clean = args
    path = os.path.join(FEAT_DIR, "dfl_{}_r{}.npz".format(match_id, realisation))
    if os.path.exists(path):
        return path
    rng = np.random.default_rng([SEED, data.DFL_MATCHES.index(match_id), realisation])
    clean_match = data.load_dfl(match_id)
    noisy = noise.apply(clean_match, stats, clean, rng)
    names, xa = features.compute(noisy)
    _, xb = features.compute(noisy, swap=True)
    target = decode.controlled_labels(clean_match.labels, clean_match.segments, MIN_SPELL)
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.savez(path, xa=xa, xb=xb, y=target, raw=clean_match.labels,
             segments=np.array(clean_match.segments), names=np.array(names))
    return path


def build_features():
    stats, clean = _noise_stats()
    jobs = [(m, r, stats, clean) for m in data.DFL_MATCHES for r in range(N_TRAIN_NOISE + 1)]
    with Pool(7) as pool:
        pool.map(_game_features, jobs)


_SIM_TAGS = {}


def sim_tags(match_id, realisation):
    """Simulated Smart-Tagging set pieces for one noise realisation of a DFL game (seeded)."""
    from . import tags
    key = (match_id, realisation)
    if key not in _SIM_TAGS:
        with open(TAG_STATS) as handle:
            stats = json.load(handle)
        match = data.load_dfl(match_id)
        rng = np.random.default_rng([SEED, data.DFL_MATCHES.index(match_id), realisation, 99])
        _SIM_TAGS[key] = tags.simulate(tags.dfl_restarts(match, match_id), stats,
                                       match.segments, rng)
    return _SIM_TAGS[key]


def add_tag_features(game, tag_list):
    from . import tags
    n = len(game["xa"])
    segs = [tuple(s) for s in game["segments"]]
    names, ta = tags.features(n, segs, tag_list)
    _, tb = tags.features(n, segs, tag_list, swap=True)
    game["xa"] = np.concatenate([game["xa"], ta], axis=1)
    game["xb"] = np.concatenate([game["xb"], tb], axis=1)
    game["names"] = np.concatenate([game["names"], names])
    return game


def _load(match_id, realisation):
    z = np.load(os.path.join(FEAT_DIR, "dfl_{}_r{}.npz".format(match_id, realisation)))
    game = {k: z[k] for k in z.files}
    if USE_TAGS:
        game = add_tag_features(game, sim_tags(match_id, realisation))
    return game


def _rows(game):
    """Training rows from both orientations, every STRIDE-th labelled frame."""
    y = game["y"]
    idx = np.flatnonzero(y != data.X)[::STRIDE]
    xs = np.concatenate([game["xa"][idx], game["xb"][idx]])
    ys = np.concatenate([y[idx], features.swap_labels(y)[idx]])
    return xs, ys


def fit(match_ids):
    xs, ys = [], []
    for m in match_ids:
        for r in range(1, N_TRAIN_NOISE + 1):
            x, y = _rows(_load(m, r))
            xs.append(x)
            ys.append(y)
    ds = lgb.Dataset(np.concatenate(xs), np.concatenate(ys), free_raw_data=True)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=N_ROUNDS)


def predict_proba(model, xa, xb):
    """Test-time augmentation: average the prediction and the swapped prediction swapped back."""
    pa = model.predict(xa)
    pb = model.predict(xb)[:, [1, 0, 2]]
    return (pa + pb) / 2.0


def out_of_fold():
    """Realisation 0 of each game is held out: never trained on, only predicted."""
    oof = {}
    for held in data.DFL_MATCHES:
        t = time.time()
        model = fit([m for m in data.DFL_MATCHES if m != held])
        game = _load(held, 0)
        oof[held] = predict_proba(model, game["xa"], game["xb"])
        print("  fold {} done in {:.0f}s".format(held, time.time() - t), flush=True)
    np.savez(os.path.join(MODEL_DIR, "oof_probs.npz"), **oof)
    return oof


_TARGETS = {}


def _target(match_id):
    """(labels, segments) of a game's evaluation realisation, loaded once."""
    if match_id not in _TARGETS:
        z = np.load(os.path.join(FEAT_DIR, "dfl_{}_r0.npz".format(match_id)))
        _TARGETS[match_id] = {"y": z["y"], "segments": z["segments"]}
    return _TARGETS[match_id]


def score(games, probs, params):
    """Mean spellF1 and macroF1 over games for one decoder setting."""
    spell, macro = [], []
    for m in games:
        game = _target(m)
        segs = [tuple(s) for s in game["segments"]]
        pred = decode.decode(probs[m], segs, params)
        spell.append(pm.spell_scores(game["y"], pred, segs)["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(game["y"], pred))))
    return float(np.mean(spell)), float(np.mean(macro)), spell


_OOF = None


def _score_params(params):
    s, mf, _ = score(data.DFL_MATCHES, _OOF, params)
    return s, mf, params


def _init_pool(path):
    global _OOF
    z = np.load(path)
    _OOF = {k: z[k] for k in z.files}


GRID = {"c_switch": [2, 3, 4, 5, 6, 8, 12, 16, 24],
        "c_to_dead": [2, 4, 6, 8, 12, 16, 24, 32, 64],
        "c_from_dead": [4, 8, 16, 24, 32, 48, 64, 96, 128]}


def tune_decoder(oof_path=None, grid=GRID):
    """Exhaustive grid over the three Viterbi costs, scored on out-of-fold DFL probabilities."""
    oof_path = oof_path or os.path.join(MODEL_DIR, "oof_probs.npz")
    combos = [{"c_switch": a, "c_to_dead": b, "c_from_dead": c, "temp": 1.0}
              for a, b, c in itertools.product(*grid.values())]
    with Pool(16, initializer=_init_pool, initargs=(oof_path,)) as pool:
        results = pool.map(_score_params, combos)
    # Ties broken by grid order, so the choice is deterministic.
    return max(results, key=lambda r: r[0])


def _seq_games(match_ids):
    games = []
    for m in match_ids:
        for r in range(1, N_TRAIN_NOISE + 1):
            g = _load(m, r)
            games.append((g["xa"], g["xb"], g["y"], [tuple(s) for s in g["segments"]]))
    return games


def out_of_fold_gru():
    from . import seq
    oof = {}
    for held in data.DFL_MATCHES:
        t = time.time()
        model = seq.fit(_seq_games([m for m in data.DFL_MATCHES if m != held]))
        g = _load(held, 0)
        oof[held] = seq.predict(model, g["xa"], g["xb"], [tuple(s) for s in g["segments"]])
        print("  gru fold {} done in {:.0f}s".format(held, time.time() - t), flush=True)
    np.savez(os.path.join(MODEL_DIR, "oof_gru.npz"), **oof)
    return oof


def _cached(path, compute):
    """Out-of-fold probabilities are deterministic; reuse them if this run already made them."""
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    return compute()


def main():
    from . import seq
    os.makedirs(MODEL_DIR, exist_ok=True)
    t = time.time()
    build_features()
    print("features ready ({:.0f}s)".format(time.time() - t), flush=True)

    oof_lgb = _cached(os.path.join(MODEL_DIR, "oof_probs.npz"), out_of_fold)
    oof_gru = _cached(os.path.join(MODEL_DIR, "oof_gru.npz"), out_of_fold_gru)
    oof = {m: (oof_lgb[m] + oof_gru[m]) / 2.0 for m in data.DFL_MATCHES}
    avg_path = os.path.join(MODEL_DIR, "oof_avg.npz")
    np.savez(avg_path, **oof)

    s_raw, m_raw, _ = score(data.DFL_MATCHES, oof,
                            {"c_switch": 0, "c_to_dead": 0, "c_from_dead": 0})
    print("held-out DFL, per-frame argmax : spellF1 {:.4f}  macroF1 {:.4f}".format(s_raw, m_raw))
    s, mf, params = tune_decoder(avg_path)
    _, _, per_game = score(data.DFL_MATCHES, oof, params)
    print("held-out DFL, tuned Viterbi    : spellF1 {:.4f}  macroF1 {:.4f}  {}".format(
        s, mf, params))
    print("  per game spellF1:", " ".join("{:.3f}".format(v) for v in per_game))
    with open(os.path.join(MODEL_DIR, "decoder.json"), "w") as handle:
        json.dump(dict(params, cv_spell_f1=s, cv_macro_f1=mf), handle, indent=2)

    model = fit(data.DFL_MATCHES)
    model.save_model(os.path.join(MODEL_DIR, "lgbm.txt"))
    print("final LightGBM saved ({:.0f}s)".format(time.time() - t), flush=True)
    seq.save(seq.fit(_seq_games(data.DFL_MATCHES)), os.path.join(MODEL_DIR, "gru.pt"))
    names = list(_load(data.DFL_MATCHES[0], 0)["names"])
    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as handle:
        json.dump(names, handle)
    print("final models saved ({:.0f}s total)".format(time.time() - t))


if __name__ == "__main__":
    main()
