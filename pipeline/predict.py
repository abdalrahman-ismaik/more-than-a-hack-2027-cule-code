"""Label every frame of the provided game with the trained model and decoder.

    python -m pipeline.predict [--name NAME]     # writes submissions/NAME.csv

Uses only the provided tracking for BAR-ATH plus models/ (trained on DFL games). Frames outside
the two halves are written as X; inside them every frame gets H, A or D.
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from . import data, decode, features

ROOT = data.ROOT
MODEL_DIR = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "submissions")
CODES = np.array(["H", "A", "D", "X"])

# Organizer FAQ: the graded second half starts before the kickoff. Preserve the
# published boundary policy already present in the team's latest main branch.
KNOWN_DEAD = ((110808, 113742),)  # half-open, tactical-video frame indices


def apply_known_dead(probs):
    """Apply the organizer-published pre-kickoff interval before sequence decoding."""
    result = probs.copy()
    for start, stop in KNOWN_DEAD:
        result[start:stop] = (1e-6, 1e-6, 1.0 - 2e-6)
    return result


def probabilities(match, model, xa, xb):
    probs = np.full((match.n_frames, 3), np.nan)
    for start, stop in match.segments:
        pa = model.predict(xa[start:stop])
        pb = model.predict(xb[start:stop])[:, [1, 0, 2]]
        probs[start:stop] = (pa + pb) / 2.0
    return probs


def write_csv(path, codes):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = CODES[codes]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="CuleCode")
    args = parser.parse_args()

    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm.txt"))
    with open(os.path.join(MODEL_DIR, "decoder.json")) as handle:
        params = json.load(handle)

    match = data.load_atd()
    from . import seq
    gru = seq.load(os.path.join(MODEL_DIR, "gru.pt"))
    names, xa = features.compute(match)
    swapped_names, xb = features.compute(match, swap=True)
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as handle:
        expected_names = json.load(handle)
    if list(names) != expected_names or list(swapped_names) != expected_names:
        raise ValueError("Computed features do not match the committed model feature order")
    p_lgb = probabilities(match, model, xa, xb)
    p_gru = seq.predict(gru, xa, xb, match.segments)
    probs = (p_lgb + p_gru) / 2.0
    np.save(os.path.join(data.CACHE_DIR, "BAR-ATH_probs.npy"), probs)
    probs = apply_known_dead(np.nan_to_num(probs, nan=1 / 3))
    codes = decode.decode(probs, match.segments, params,
                          n_frames=match.n_frames)
    path = write_csv(os.path.join(OUT_DIR, args.name + ".csv"), codes)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
