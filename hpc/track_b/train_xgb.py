"""Train an XGBoost comparison model on the existing cached feature folds.

This mirrors pipeline.train's leave-one-game-out data preparation while writing only beneath
models_xgb/. Run one fold first:

    python -m pipeline.train_xgb --fold J03WMX

Then, if the fold output is sound, run all folds and fit a final model:

    python -m pipeline.train_xgb --all
"""

import argparse
import json
import os
import time

import numpy as np
import xgboost as xgb

from . import data, train

MODEL_DIR = os.environ.get("MTAH_XGB_MODEL_DIR", os.path.join(data.ROOT, "models_xgb"))
N_ROUNDS = 400
MAX_BIN = 256
PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.1,
    "max_bin": MAX_BIN,
    "seed": train.SEED,
    "nthread": 4,
}


def _feature_cache_path(match_id, realisation):
    return os.path.join(train.FEAT_DIR, "g_{}_r{}.npz".format(match_id, realisation))


def _check_cache():
    missing = [
        _feature_cache_path(match_id, realisation)
        for match_id in data.TRAIN_MATCHES
        for realisation in range(train.N_TRAIN_NOISE + 1)
        if not os.path.isfile(_feature_cache_path(match_id, realisation))
    ]
    if missing:
        raise FileNotFoundError(
            "Feature cache is incomplete ({} missing; first: {}). "
            "Do not rebuild it while the LightGBM job is running.".format(
                len(missing), missing[0]))
    sample_path = _feature_cache_path(data.TRAIN_MATCHES[0], 0)
    with np.load(sample_path) as sample:
        feature_count = int(sample["xa"].shape[1])
        names_count = len(sample["names"])
        if feature_count != names_count:
            raise ValueError("Cached feature matrix/name count mismatch: {} vs {}".format(
                feature_count, names_count))
    print("feature cache ready: {} files, {} features".format(
        len(data.TRAIN_MATCHES) * (train.N_TRAIN_NOISE + 1), feature_count), flush=True)


def _fit(match_ids):
    xs, ys = [], []
    for match_id in match_ids:
        for realisation in range(1, train.N_TRAIN_NOISE + 1):
            x, y = train._rows(train._load(match_id, realisation))
            xs.append(x)
            ys.append(y)

    x_train = np.ascontiguousarray(np.concatenate(xs), dtype=np.float32)
    y_train = np.ascontiguousarray(np.concatenate(ys), dtype=np.int32)
    del xs, ys
    print("fit rows={} features={} labels={}".format(
        len(y_train), x_train.shape[1], np.bincount(y_train, minlength=3).tolist()), flush=True)

    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=MAX_BIN)
    del x_train, y_train
    return xgb.train(PARAMS, dtrain, num_boost_round=N_ROUNDS, verbose_eval=50)


def _predict(model, match_id):
    game = train._load(match_id, 0)
    da = xgb.DMatrix(np.ascontiguousarray(game["xa"], dtype=np.float32))
    db = xgb.DMatrix(np.ascontiguousarray(game["xb"], dtype=np.float32))
    pa = model.predict(da)
    pb = model.predict(db)[:, [1, 0, 2]]
    return ((pa + pb) / 2.0).astype(np.float32)


def _report_fold(held_out, probabilities):
    # Read the frozen baseline only; the active LightGBM retrain writes under models/.
    baseline_dir = os.path.join(data.ROOT, "models_committed")
    decoder_path = os.path.join(baseline_dir, "decoder.json")
    if not os.path.isfile(decoder_path):
        print("No saved decoder.json; skipping spellF1 comparison", flush=True)
        return
    try:
        with open(decoder_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        params = {key: saved[key] for key in
                  ("c_switch", "c_to_dead", "c_from_dead", "temp") if key in saved}
        xgb_scores = train.score([held_out], {held_out: probabilities}, params)
        print("XGBoost fold={} spellF1={:.4f} macroF1={:.4f} using saved decoder".format(
            held_out, xgb_scores[0], xgb_scores[1]), flush=True)

    except (OSError, ValueError, KeyError, EOFError) as exc:
        print("Fold trained; comparison skipped because existing model outputs "
              "were unavailable or changing: {}".format(exc), flush=True)


def run_fold(held_out):
    if held_out not in data.TRAIN_MATCHES:
        raise ValueError("Unknown held-out match: {}".format(held_out))
    _check_cache()
    os.makedirs(MODEL_DIR, exist_ok=True)
    started = time.time()
    train_ids = [m for m in data.TRAIN_MATCHES if m != held_out]
    model = _fit(train_ids)
    probabilities = _predict(model, held_out)

    pred_path = os.path.join(MODEL_DIR, "oof_fold_{}.npz".format(held_out))
    np.savez_compressed(pred_path, **{held_out: probabilities})
    model_path = os.path.join(MODEL_DIR, "xgb_fold_{}.json".format(held_out))
    model.save_model(model_path)
    print("fold={} rows={} saved={} model={} elapsed={:.1f}s".format(
        held_out, len(probabilities), pred_path, model_path, time.time() - started), flush=True)
    _report_fold(held_out, probabilities)


def run_all():
    _check_cache()
    os.makedirs(MODEL_DIR, exist_ok=True)
    partial_path = os.path.join(MODEL_DIR, "oof_xgb_partial.npz")
    oof = {}
    if os.path.isfile(partial_path):
        with np.load(partial_path) as cached:
            oof = {key: cached[key] for key in cached.files}
        unexpected = set(oof) - set(data.TRAIN_MATCHES)
        if unexpected:
            raise ValueError("Unexpected cached fold keys: {}".format(sorted(unexpected)))

    for held_out in data.TRAIN_MATCHES:
        fold_path = os.path.join(MODEL_DIR, "oof_fold_{}.npz".format(held_out))
        if held_out in oof or not os.path.isfile(fold_path):
            continue
        with np.load(fold_path) as cached:
            if held_out in cached.files:
                oof[held_out] = cached[held_out]
        if held_out in oof:
            print("fold={} reusing completed sanity fold".format(held_out), flush=True)
    if oof:
        temp_path = partial_path + ".tmp"
        with open(temp_path, "wb") as handle:
            np.savez_compressed(handle, **oof)
        os.replace(temp_path, partial_path)

    for held_out in data.TRAIN_MATCHES:
        if held_out in oof:
            print("fold={} already cached; skipping".format(held_out), flush=True)
            continue
        started = time.time()
        model = _fit([m for m in data.TRAIN_MATCHES if m != held_out])
        oof[held_out] = _predict(model, held_out)
        temp_path = partial_path + ".tmp"
        with open(temp_path, "wb") as handle:
            np.savez_compressed(handle, **oof)
        os.replace(temp_path, partial_path)
        print("fold={} done in {:.1f}s; {}/{} folds saved".format(
            held_out, time.time() - started, len(oof), len(data.TRAIN_MATCHES)), flush=True)

    final_oof_path = os.path.join(MODEL_DIR, "oof_xgb.npz")
    np.savez_compressed(final_oof_path, **oof)
    print("all OOF folds saved: {}".format(final_oof_path), flush=True)

    started = time.time()
    final_model = _fit(data.TRAIN_MATCHES)
    final_model_path = os.path.join(MODEL_DIR, "xgb.json")
    final_model.save_model(final_model_path)
    print("final model saved: {} ({:.1f}s)".format(
        final_model_path, time.time() - started), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", help="Train and predict one leave-one-game-out fold")
    group.add_argument("--all", action="store_true", help="Run all OOF folds and final fit")
    args = parser.parse_args()

    # Fail before starting if the active runtime cannot see a CUDA-enabled XGBoost build.
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError("This XGBoost package was built without CUDA support")
    print("xgboost={} cuda={} model_dir={}".format(
        xgb.__version__, xgb.build_info().get("USE_CUDA"), MODEL_DIR), flush=True)
    if args.all:
        run_all()
    else:
        run_fold(args.fold)


if __name__ == "__main__":
    main()
