"""Quick version of the Smart Tagging idea: fixed rules on the model's probabilities.

    python -m pipeline.tag_rules        # tunes the rules on held-out DFL games, writes tag_rules.json

The rules (tags.apply_rules) are applied to the out-of-fold DFL probabilities with simulated tags
(realisation 0, the held-out one), and their four settings are chosen by spellF1 with the decoder
fixed. The combination with every multiplier at 1 is the model without tags, so the table always
contains the baseline it has to beat.
"""

import itertools
import json
import os
from multiprocessing import Pool

import numpy as np

from . import data, tags, train

RULE_GRID = {"lead": [50, 75, 117, 150],
             "k_dead": [1, 1.5, 2, 4, 8],
             "k_team": [1, 1.5, 2, 4],
             "w_team": [50, 125, 250]}

_OOF, _TAGS, _DEC = None, None, None


def _init(oof_path, decoder):
    global _OOF, _TAGS, _DEC
    z = np.load(oof_path)
    _OOF = {k: z[k] for k in z.files}
    _TAGS = {m: train.sim_tags(m, 0) for m in data.DFL_MATCHES}
    _DEC = decoder


def _score(params):
    probs = {m: tags.apply_rules(_OOF[m], _TAGS[m], params) for m in data.DFL_MATCHES}
    s, mf, per_game = train.score(data.DFL_MATCHES, probs, _DEC)
    return s, mf, per_game, params


def tune(model_dir=train.BASE_MODEL_DIR):
    oof_path = os.path.join(model_dir, "oof_avg.npz")
    with open(os.path.join(model_dir, "decoder.json")) as handle:
        dec = json.load(handle)
    decoder = {k: dec[k] for k in ("c_switch", "c_to_dead", "c_from_dead", "temp")}
    combos = [dict(zip(RULE_GRID, v)) for v in itertools.product(*RULE_GRID.values())]
    with Pool(16, initializer=_init, initargs=(oof_path, decoder)) as pool:
        results = pool.map(_score, combos)
    base = next(r for r in results if r[3]["k_dead"] == 1 and r[3]["k_team"] == 1)
    best = max(results, key=lambda r: r[0])
    print("no tags      : spellF1 {:.4f}  macroF1 {:.4f}".format(base[0], base[1]))
    print("best rules   : spellF1 {:.4f}  macroF1 {:.4f}  {}".format(best[0], best[1], best[3]))
    print("  per game no tags :", " ".join("{:.3f}".format(v) for v in base[2]))
    print("  per game rules   :", " ".join("{:.3f}".format(v) for v in best[2]))
    top = sorted(results, key=lambda r: -r[0])[:8]
    for r in top:
        print("   {:.4f} {:.4f} {}".format(r[0], r[1], r[3]))
    out = dict(best[3], cv_spell_f1=best[0], cv_macro_f1=best[1], cv_spell_f1_no_tags=base[0],
               cv_macro_f1_no_tags=base[1], per_game=best[2], per_game_no_tags=base[2])
    with open(os.path.join(model_dir, "tag_rules.json"), "w") as handle:
        json.dump(out, handle, indent=2)
    return out


if __name__ == "__main__":
    tune()
