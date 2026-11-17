"""Extra GRU seeds: out-of-fold predictions and final fits for a seed ensemble.

    python -m pipeline.gru_seeds oof SEED      # models/possession/oof_gru_s<SEED>.npz
    python -m pipeline.gru_seeds final SEED    # models/possession/gru_s<SEED>.pt

Same data, features and settings as the committed GRU (seq.SEED = 2027); only the seed differs.
Dropped experiment: averaging seeds 2027 and 1 gave held-out spellF1 0.6895 vs 0.6900.
"""

import os
import sys
import time

import numpy as np

from . import data, seq, train


def oof(seed):
    out = {}
    for held in data.DFL_MATCHES:
        t = time.time()
        model = seq.fit(train._seq_games([m for m in data.DFL_MATCHES if m != held]), seed=seed)
        g = train._load(held, 0)
        out[held] = seq.predict(model, g["xa"], g["xb"], [tuple(s) for s in g["segments"]])
        print("  gru seed {} fold {} done in {:.0f}s".format(seed, held, time.time() - t), flush=True)
    np.savez(os.path.join(train.MODEL_DIR, "oof_gru_s{}.npz".format(seed)), **out)


def final(seed):
    model = seq.fit(train._seq_games(data.DFL_MATCHES), seed=seed)
    seq.save(model, os.path.join(train.MODEL_DIR, "gru_s{}.pt".format(seed)))


if __name__ == "__main__":
    {"oof": oof, "final": final}[sys.argv[1]](int(sys.argv[2]))
