"""Validate the decoder on the Metrica sample matches (other games, so allowed for tuning).

The Metrica sample data has tracking with the ball and an event log, but no possession labels.
The ground truth used here is therefore a PROXY, derived from the event log by derive_labels():

- PASS, SHOT, RECOVERY and SET PIECE events give the team in possession from their start onward.
- BALL OUT and FAULT RECEIVED mean the ball is dead until the next possession event.
- Other events (challenges, ball lost, cards) leave the current state unchanged.

It is a proxy, not an answer key. Use it to compare settings of the decoder with each other, not
as a number to report as the official score. The scorer functions are the official ones.

    python -m src.metrica_val                 # scores the default weights on games 1 and 2
"""

import os

import numpy as np
import pandas as pd

from src import decode
from src.atd import AWAY, HOME
from src.config import FPS, ROOT

GAME_DIR = os.path.join(ROOT, "external", "metrica-sample-data", "data")
POSSESSION_TYPES = {"PASS", "SHOT", "RECOVERY", "SET PIECE"}
DEAD_TYPES = {"BALL OUT", "FAULT RECEIVED"}


def load_game(number):
    """xy (n, 28, 2), ball (n, 2), slot_team, events, graded spans (list of (start, stop_exclusive))."""
    folder = os.path.join(GAME_DIR, "Sample_Game_{}".format(number))
    prefix = os.path.join(folder, "Sample_Game_{}".format(number))
    home = pd.read_csv(prefix + "_RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(prefix + "_RawTrackingData_Away_Team.csv", skiprows=2)
    events = pd.read_csv(prefix + "_RawEventsData.csv")

    n = int(max(home["Frame"].max(), away["Frame"].max()))
    frame_index = home["Frame"].to_numpy(int) - 1
    ball_cols = home.iloc[:, -2:].to_numpy(np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[frame_index] = ball_cols

    def players(table):
        values = table.iloc[:, 3:-2].to_numpy(np.float32)
        out = np.full((n, values.shape[1] // 2, 2), np.nan, dtype=np.float32)
        out[frame_index] = values.reshape(len(table), -1, 2)
        return out

    home_xy, away_xy = players(home), players(away)
    xy = np.concatenate([home_xy, away_xy], axis=1)
    slot_team = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], dtype=np.int8)

    spans = []
    for period in sorted(events["Period"].dropna().unique()):
        rows = events[events["Period"] == period]
        spans.append((int(rows["Start Frame"].min()) - 1, int(rows["Start Frame"].max())))
    return xy, ball, slot_team, events, spans


def derive_labels(events, n, spans):
    """Proxy H/A/D/X labels from the event log. X outside the event span of each period."""
    labels = np.full(n, "X", dtype="<U1")
    for period, (lo, hi) in zip(sorted(events["Period"].dropna().unique()), spans):
        rows = events[events["Period"] == period].sort_values("Start Frame", kind="stable")
        starts = rows["Start Frame"].to_numpy(int) - 1
        state = "D"
        for i, (_, event) in enumerate(rows.iterrows()):
            kind = str(event["Type"]).strip().upper()
            team = str(event["Team"]).strip().lower()
            if kind in DEAD_TYPES:
                state = "D"
            elif kind in POSSESSION_TYPES and team in ("home", "away"):
                state = "H" if team == "home" else "A"
            end = starts[i + 1] if i + 1 < len(starts) else hi + 1
            labels[max(0, starts[i]): min(n, end)] = state
    return labels


def decode_game(xy, ball, slot_team, spans, params=None):
    """Run the decoder over each period span and return labels with X outside the spans."""
    from src.features import control_evidence

    params = params or {}
    control, out_of_play = control_evidence(xy, ball, slot_team)
    transitions = decode.make_transitions(**params.get("transitions", {}))
    weights = params.get("weights", {})
    zeros = np.zeros(len(control), dtype=bool)
    labels = np.full(len(control), "X", dtype="<U1")
    for lo, hi in spans:
        span = slice(lo, hi)
        ll = decode.emissions(control[span], out_of_play[span], zeros[span], zeros[span], zeros[span],
                              **weights)
        path = decode.viterbi(ll, transitions)
        labels[span] = np.array(["H", "A", "D"])[path]
    return labels


def score(gt_labels, pred_labels):
    """The official scorer's metrics for one submission."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "challenge", "evaluation"))
    import possession_metrics as pm

    return pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels))


if __name__ == "__main__":
    for number in (1, 2):
        xy, ball, slot_team, events, spans = load_game(number)
        gt = derive_labels(events, len(ball), spans)
        pred = decode_game(xy, ball, slot_team, spans)
        graded = gt != "X"
        print("Game {}: {} proxy-labelled frames, H {} A {} D {}".format(
            number, int(graded.sum()), int((gt == "H").sum()), int((gt == "A").sum()),
            int((gt == "D").sum())))
        result = score(gt, pred)
        print({k: result[k] for k in result if k in ("macro_f1", "spell_f1_mean")}
              if isinstance(result, dict) else result)
