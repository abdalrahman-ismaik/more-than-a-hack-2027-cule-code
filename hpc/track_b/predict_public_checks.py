"""Generate public-window sanity submissions for the retrained model variants.

This uses the public five-minute labels only in the separate official evaluation step. It does
not tune models, decoder settings, features, or blend weights.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from . import data, decode, features, seq
from .predict import probabilities as lgb_probabilities, write_csv

ROOT = data.ROOT
OUT_DIR = os.path.join(ROOT, "evaluation", "public_model_checks")


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _xgb_probabilities(model, xa, xb, segments, n_frames):
    probs = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for start, stop in segments:
        da = xgb.DMatrix(np.ascontiguousarray(xa[start:stop], dtype=np.float32))
        db = xgb.DMatrix(np.ascontiguousarray(xb[start:stop], dtype=np.float32))
        pa = model.predict(da)
        pb = model.predict(db)[:, [1, 0, 2]]
        probs[start:stop] = (pa + pb) / 2.0
    return probs


def _write_variant(name, probs, segments, decoder_path):
    decoder_params = _load_json(decoder_path)
    labels = decode.decode(np.nan_to_num(probs, nan=1.0 / 3.0), segments,
                           decoder_params, n_frames=len(probs))
    return write_csv(os.path.join(OUT_DIR, name + ".csv"), labels)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    match = data.load_atd()
    _, xa = features.compute(match)
    _, xb = features.compute(match, swap=True)

    lgb_model = lgb.Booster(model_file=os.path.join(ROOT, "models", "lgbm.txt"))
    p_lgb = lgb_probabilities(match, lgb_model, xa, xb)
    gru_model = seq.load(os.path.join(ROOT, "models", "gru.pt"))
    p_gru = seq.predict(gru_model, xa, xb, match.segments)
    p_lgb_gru = (p_lgb + p_gru) / 2.0

    xgb_model = xgb.Booster()
    xgb_model.load_model(os.path.join(ROOT, "models_xgb", "xgb.json"))
    p_xgb = _xgb_probabilities(xgb_model, xa, xb, match.segments, match.n_frames)

    variants = [
        ("retrained_lgbm_gru", p_lgb_gru,
         os.path.join(ROOT, "models", "decoder.json")),
        ("retrained_xgb", p_xgb,
         os.path.join(ROOT, "models_xgb", "decoder_xgb.json")),
        ("retrained_xgb_gru", (p_xgb + p_gru) / 2.0,
         os.path.join(ROOT, "models_xgb", "decoder_xgb_gru.json")),
        ("retrained_all_three", (p_lgb + p_gru + p_xgb) / 3.0,
         os.path.join(ROOT, "models_xgb", "decoder_ensemble.json")),
    ]
    for name, probs, decoder_path in variants:
        path = _write_variant(name, probs, match.segments, decoder_path)
        print("wrote", os.path.relpath(path, ROOT), flush=True)


if __name__ == "__main__":
    main()
