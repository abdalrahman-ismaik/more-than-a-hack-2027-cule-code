"""Choose decoder settings on half of the SkillCorner matches and check on the other half.

Only other-match data is used here (allowed). The labels are the official phases-of-play
possession spans, not a proxy. The grid values are the same nominal rates as sweep_metrica
(set for 25 fps); they are scaled to 10 fps before decoding.

    python -m src.sweep_skillcorner
"""

import itertools
import os

import numpy as np

from src import skillcorner_val

GRID = {
    "w_control": [0.5, 1.0, 2.0],
    "bias": [0.0, 0.5, 1.0],
    "p_team_to_dead": [0.0005, 0.0015, 0.004],
    "p_switch": [0.0002, 0.0006, 0.002],
    "p_dead_to_team": [0.002, 0.004, 0.01],
}

# 25 fps -> 10 fps: each SkillCorner frame spans 2.5x longer, so rates scale by 2.5.
FPS_SCALE = 2.5

DEFAULTS = {"w_control": 0.5, "bias": 0.0, "p_team_to_dead": 0.004,
            "p_switch": 0.002, "p_dead_to_team": 0.004}


def params_from(values):
    return {
        "weights": {"w_control": values["w_control"], "bias": values["bias"]},
        "transitions": {"p_team_to_dead": values["p_team_to_dead"] * FPS_SCALE,
                        "p_switch": values["p_switch"] * FPS_SCALE,
                        "p_dead_to_team": values["p_dead_to_team"] * FPS_SCALE},
    }


def main():
    ids = sorted(int(d) for d in os.listdir(os.path.join(skillcorner_val.DATA, "matches")))
    select_ids, check_ids = ids[: len(ids) // 2], ids[len(ids) // 2:]
    print("select on {} matches, check on {}".format(len(select_ids), len(check_ids)))

    games = {}
    for mid in ids:
        games[mid] = skillcorner_val.load_match(mid)

    def run(mid, values):
        gt, spans, xy, ball, slot_team, _ = games[mid]
        pred = skillcorner_val.decode_game(xy, ball, slot_team, spans, params_from(values))
        result = skillcorner_val.score(gt, pred)
        return result["spell_f1_mean"], result["macro_f1"]

    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        values = dict(zip(keys, combo))
        spells, macros = zip(*(run(mid, values) for mid in select_ids))
        rows.append((float(np.mean(spells)), float(np.mean(macros)), values))
    rows.sort(key=lambda r: (-r[0], -r[1]))

    print("top 8 on the selection half:")
    for spell, macro, values in rows[:8]:
        print("  spell {:.4f}  macro {:.4f}  {}".format(spell, macro, values))

    for name, values in (("best", rows[0][2]), ("current defaults", DEFAULTS)):
        spells, macros = zip(*(run(mid, values) for mid in check_ids))
        print("\n{} checked on the other half:".format(name))
        print("  spell {:.4f}  macro {:.4f}".format(np.mean(spells), np.mean(macros)))
        print("  {}".format(values))


if __name__ == "__main__":
    main()
