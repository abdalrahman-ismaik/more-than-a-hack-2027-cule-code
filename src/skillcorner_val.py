"""Validate the decoder on the SkillCorner open data (20 A-League matches, other games).

Source: github.com/SkillCorner/opendata (MIT licence). Broadcast tracking at 10 fps with the
ball, plus a published phases-of-play file. Each phase row is an in-play span with the team in
possession; frames between phases are out of play. These are official labels for OTHER matches,
so they are allowed for tuning, and unlike the Metrica event proxy they are not derived by us.

Differences from the challenge match: 10 fps (transition rates scaled by 25/10), coordinates in
metres centred on the pitch (converted to the normalised [0, 1] form our features expect).

Results are cached in the repo's cache/ folder (git-ignored) so sweeps are fast.

    python -m src.skillcorner_val               # all matches, current default weights
    python -m src.skillcorner_val 1925299 1996435
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from src import decode
from src.atd import AWAY, HOME
from src.metrica_val import decode_game, score

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "external", "skillcorner-opendata", "data")
CACHE = os.path.join(DATA, "cache")

# Per-frame transition rates were chosen for 25 fps tracking; SkillCorner is 10 fps, so each
# frame covers 2.5x longer. Rates scale with time: 0.004/s * 2.5 = 0.01 per frame here.
SCALED_TRANSITIONS = {"p_team_to_dead": 0.01, "p_switch": 0.005, "p_dead_to_team": 0.01}
DEFAULT_PARAMS = {"transitions": SCALED_TRANSITIONS, "weights": {}}


def load_match(match_id):
    """Returns gt labels, decoder params, and the inputs decode_game needs (spans, xy, ball, slot_team).

    gt is H/A/D inside periods 1 and 2 (D in the gaps between in-play phases) and X outside.
    Also returns the per-frame possession field for cross-checks.
    """
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "{}.npz".format(match_id))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["gt"].astype("<U1"), z["spans"], z["xy"], z["ball"], z["slot_team"], z["poss"].astype("<U1")

    folder = os.path.join(DATA, "matches", str(match_id))
    meta = json.load(open(os.path.join(folder, str(match_id) + "_match.json"), encoding="utf-8"))
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    length, width = meta["pitch_length"], meta["pitch_width"]
    players = [p for p in meta["players"] if p["team_id"] in (home_id, away_id)]
    slot_of = {p["id"]: i for i, p in enumerate(players)}
    slot_team = np.array([HOME if p["team_id"] == home_id else AWAY for p in players], dtype=np.int8)

    frames, periods, poss, ball_rows, xy_rows = [], [], [], [], []
    with open(os.path.join(folder, str(match_id) + "_tracking_extrapolated.jsonl"), encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            frames.append(int(fr["frame"]))
            periods.append(fr["period"])
            group = fr["possession"].get("group")
            poss.append("H" if group == "home team" else "A" if group == "away team" else "X")
            b = fr["ball_data"]
            ball_rows.append((b.get("x"), b.get("y")))
            row = np.full((len(players), 2), np.nan, dtype=np.float32)
            for p in fr["player_data"]:
                j = slot_of.get(p["player_id"])
                if j is not None and p.get("x") is not None:
                    row[j] = (p["x"], p["y"])
            xy_rows.append(row)

    frame_to_row = np.full(max(frames) + 1, -1, dtype=np.int64)
    frame_to_row[np.array(frames)] = np.arange(len(frames))
    ball_m = np.array(ball_rows, dtype=np.float32)
    # Metres centred on the pitch -> normalised [0, 1], the form our features use.
    ball = np.stack([ball_m[:, 0] / length + 0.5, ball_m[:, 1] / width + 0.5], axis=1).astype(np.float32)
    xy_m = np.stack(xy_rows)
    xy = np.stack([(xy_m[..., 0] / length + 0.5).astype(np.float32),
                   (xy_m[..., 1] / width + 0.5).astype(np.float32)], axis=-1)

    n = len(frames)
    frames_arr = np.array(frames)
    periods_arr = np.array([p if p is not None else 0 for p in periods])
    spans = []
    for period in (1, 2):
        rows = frames_arr[periods_arr == period]
        if len(rows):
            spans.append((int(rows.min()), int(rows.max()) + 1))

    gt = _labels_from_phases(match_id, home_id, frame_to_row, n, spans)
    np.savez_compressed(cache, gt=gt, spans=np.array(spans), xy=xy, ball=ball,
                        slot_team=slot_team, poss=np.array(poss, dtype="<U1"))
    return gt, spans, xy, ball, slot_team, np.array(poss, dtype="<U1")


def _labels_from_phases(match_id, home_id, frame_to_row, n, spans):
    phases = pd.read_csv(os.path.join(DATA, "matches", str(match_id),
                                      str(match_id) + "_phases_of_play.csv"), encoding="utf-8")
    labels = np.full(n, "X", dtype="<U1")
    for span in spans:
        labels[span[0]:span[1]] = "D"  # in-play default; phases overwrite with H/A
    for _, row in phases.iterrows():
        if row["period"] not in (1, 2):
            continue
        lo, hi = frame_to_row[row["frame_start"]], frame_to_row[row["frame_end"]]
        if lo < 0 or hi < 0:
            continue
        team = "H" if row["team_in_possession_id"] == home_id else "A"
        labels[lo:hi] = team
    return labels


def run_match(match_id, params=None):
    gt, spans, xy, ball, slot_team, poss = load_match(match_id)
    pred = decode_game(xy, ball, slot_team, spans, params or DEFAULT_PARAMS)
    result = score(gt, pred)
    gt_poss = sum(1 for a, b in zip(gt, gt[1:]) if a in "HA" and b != a)
    pred_poss = sum(1 for a, b in zip(pred, pred[1:]) if a in "HA" and b != a)
    return match_id, result, gt_poss, pred_poss, int((gt != "X").sum())


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]]
    if not ids:
        ids = sorted(int(d) for d in os.listdir(os.path.join(DATA, "matches")))
    totals = []
    for mid in ids:
        mid, result, gt_poss, pred_poss, n_graded = run_match(mid)
        totals.append((result["macro_f1"], result["spell_f1_mean"]))
        print("{}: graded {} frames, GT {} poss vs pred {}, macroF1 {:.3f}, spellF1 {:.3f}".format(
            mid, n_graded, gt_poss, pred_poss, result["macro_f1"], result["spell_f1_mean"]))
    if len(totals) > 1:
        print("mean: macroF1 {:.3f}, spellF1 {:.3f}".format(
            np.mean([t[0] for t in totals]), np.mean([t[1] for t in totals])))
