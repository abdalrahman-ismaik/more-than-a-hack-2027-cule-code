"""Label every frame of the provided game with the trained model and decoder.

    python -m pipeline.predict [--name NAME]     # writes submissions/NAME.csv (default CuleCode)

Uses only the provided tracking for BAR-ATH plus models/ (trained on DFL games). Frames outside
the two halves are written as X; inside them every frame gets H, A or D.
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np

from . import data, decode, features, tags

ROOT = data.ROOT
MODEL_DIR = os.path.join(ROOT, "models", "possession")
if os.environ.get("MTAH_TAGS", "0") == "1":
    MODEL_DIR = os.path.join(ROOT, "models", "possession_tags")
OUT_DIR = os.path.join(ROOT, "submissions")
CODES = np.array(["H", "A", "D", "X"])

# FAQ.md: "the second half is graded from frame 110,808 but does not kick off until 113,742. Those
# 2,934 frames are graded and all of them are D." A published fact about the frame range, applied
# to the probabilities before decoding so the kickoff that follows is decoded normally.
KNOWN_DEAD = [(110808, 113742)]


def apply_known_dead(probs):
    probs = probs.copy()
    for start, stop in KNOWN_DEAD:
        probs[start:stop] = (1e-6, 1e-6, 1.0 - 2e-6)
    return probs


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
    parser.add_argument("--rules", action="store_true",
                        help="apply the quick-version tag rules from tag_rules.json")
    args = parser.parse_args()

    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm.txt"))
    with open(os.path.join(MODEL_DIR, "decoder.json")) as handle:
        params = json.load(handle)

    match = data.load_atd()
    from . import seq
    gru = seq.load(os.path.join(MODEL_DIR, "gru.pt"))
    _, xa = features.compute(match)
    _, xb = features.compute(match, swap=True)
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as handle:
        uses_tags = any(n.startswith("tag_") for n in json.load(handle))
    smart = tags.smart_restarts()
    if uses_tags:
        _, ta = tags.features(match.n_frames, match.segments, smart)
        _, tb = tags.features(match.n_frames, match.segments, smart, swap=True)
        xa = np.concatenate([xa, ta], axis=1)
        xb = np.concatenate([xb, tb], axis=1)
    p_lgb = probabilities(match, model, xa, xb)
    p_gru = seq.predict(gru, xa, xb, match.segments)
    probs = (p_lgb + p_gru) / 2.0
    np.save(os.path.join(data.CACHE_DIR, "BAR-ATH_probs.npy"), probs)
    probs = np.nan_to_num(probs, nan=1 / 3)
    if args.rules:
        with open(os.path.join(MODEL_DIR, "tag_rules.json")) as handle:
            probs = tags.apply_rules(probs, smart, json.load(handle))
    probs = apply_known_dead(probs)
    codes = decode.decode(np.nan_to_num(probs, nan=1 / 3), match.segments, params,
                          n_frames=match.n_frames)
    path = write_csv(os.path.join(OUT_DIR, args.name + ".csv"), codes)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
