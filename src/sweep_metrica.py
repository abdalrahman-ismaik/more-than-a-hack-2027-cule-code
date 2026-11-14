"""Choose decoder settings on Metrica game 1 and check the choice on game 2.

Only other-match data is used here (allowed). The labels are the event-derived proxy from
metrica_val.derive_labels. The best setting on game 1 is then scored on game 2, which it never saw.

    python -m src.sweep_metrica
"""

import itertools

from src import metrica_val

GRID = {
    "w_control": [0.5, 1.0, 2.0],
    "bias": [0.0, 0.5, 1.0],
    "p_team_to_dead": [0.0005, 0.0015, 0.004],
    "p_switch": [0.0002, 0.0006, 0.002],
    "p_dead_to_team": [0.002, 0.004, 0.01],
}


def params_from(values):
    return {
        "weights": {"w_control": values["w_control"], "bias": values["bias"]},
        "transitions": {"p_team_to_dead": values["p_team_to_dead"],
                        "p_switch": values["p_switch"],
                        "p_dead_to_team": values["p_dead_to_team"]},
    }


def main():
    games = {}
    for number in (1, 2):
        xy, ball, slot_team, events, spans = metrica_val.load_game(number)
        games[number] = (xy, ball, slot_team, spans,
                         metrica_val.derive_labels(events, len(ball), spans))

    def run(number, values):
        xy, ball, slot_team, spans, gt = games[number]
        pred = metrica_val.decode_game(xy, ball, slot_team, spans, params_from(values))
        result = metrica_val.score(gt, pred)
        return result["spell_f1_mean"], result["macro_f1"]

    keys = list(GRID)
    rows = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        values = dict(zip(keys, combo))
        spell1, macro1 = run(1, values)
        rows.append((spell1, macro1, values))
    rows.sort(key=lambda r: -r[0])

    print("top 8 on game 1 (spellF1 first, macroF1 as tie-break):")
    for spell1, macro1, values in rows[:8]:
        print("  spell {:.4f}  macro {:.4f}  {}".format(spell1, macro1, values))

    best = rows[0][2]
    spell2, macro2 = run(2, best)
    print("\nchosen on game 1, checked on game 2:")
    print("  spell {:.4f}  macro {:.4f}  {}".format(spell2, macro2, best))

    default = {"w_control": 0.5, "bias": 0.0, "p_team_to_dead": 0.004,
               "p_switch": 0.002, "p_dead_to_team": 0.004}
    spell_d1, macro_d1 = run(1, default)
    spell_d2, macro_d2 = run(2, default)
    print("\ncurrent defaults (src/decode.py):")
    print("  game 1 spell {:.4f} macro {:.4f}".format(spell_d1, macro_d1))
    print("  game 2 spell {:.4f} macro {:.4f}".format(spell_d2, macro_d2))


if __name__ == "__main__":
    main()
