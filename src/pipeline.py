"""Generate the possession file end to end: tracking + Smart Tagging -> frame,label CSV.

    python -m src.pipeline --name CuleCode_v1

Writes submissions/<name>.csv and prints the possession counts so they can be sanity-checked.
Run evaluation/validate_submission.py on the output before uploading anything.
"""

import argparse
import os
from itertools import groupby

import numpy as np
import pandas as pd

from src import atd, config, decode, features, tags

LABELS = np.array(["H", "A", "D"])

# Decoder settings, all chosen on other matches. "skillcorner" is the optimum of the sweep on
# 10 SkillCorner A-League matches (official possession labels), checked on the other 10
# (spellF1 0.4118 vs 0.3928 for default). Rates are the 25 fps equivalents; skillcorner_val
# scales them for its 10 fps data.
PRESETS = {
    "default": {},
    "skillcorner": {
        "weights": {"w_control": 1.0, "bias": 0.0},
        "transitions": {"p_team_to_dead": 0.016, "p_switch": 0.001, "p_dead_to_team": 0.004},
    },
}


def fill_ball_from_video(ball, detection_files):
    """Use the video detections where the tracking ball is missing. Tracking is kept where present."""
    filled = ball.copy()
    for path in detection_files:
        table = pd.read_csv(path)
        frames = table["frame"].to_numpy()
        found = table["det_x"].notna().to_numpy()
        gap = found & ~np.isfinite(filled[frames, 0])
        rows = frames[gap]
        filled[rows, 0] = table["det_x"].to_numpy()[gap]
        filled[rows, 1] = table["det_y"].to_numpy()[gap]
    return filled


def build_labels(n_frames=config.N_FRAMES, verbose=True, video_ball=None, params=None):
    params = params or {}
    slot_team = atd.slot_teams(config.ATD_META)
    if verbose:
        print("loading tracking ...")
    xy, ball = atd.load(config.ATD_RAW, n_frames, len(slot_team))
    if verbose:
        visible = np.isfinite(ball[:, 0]).mean() * 100
        print("  ball visible on {:.1f}% of frames (tracking)".format(visible))
    if video_ball:
        ball = fill_ball_from_video(ball, video_ball)
        if verbose:
            visible = np.isfinite(ball[:, 0]).mean() * 100
            print("  ball visible on {:.1f}% of frames (with video detections)".format(visible))

    control, out_of_play = features.control_evidence(xy, ball, slot_team)
    restart, has_home, has_away = tags.frame_evidence(
        tags.load(config.SMART_TAGGING, config.FPS), n_frames)

    labels = np.full(n_frames, "X", dtype="<U1")
    for lo, hi in config.GRADED_HALVES:
        span = slice(lo, hi + 1)
        forced = np.zeros(hi - lo + 1, dtype=bool)
        k_lo, k_hi = config.SECOND_HALF_KICKOFF_WAIT
        if lo <= k_lo <= hi:
            forced[k_lo - lo: k_hi - lo + 1] = True
        ll = decode.emissions(control[span], out_of_play[span], restart[span],
                              has_home[span], has_away[span], forced_dead=forced,
                              **params.get("weights", {}))
        transitions = decode.make_transitions(**params.get("transitions", {}))
        path = decode.viterbi(ll, transitions)
        labels[span] = LABELS[path]
        if verbose:
            print("  decoded frames {}..{}".format(lo, hi))
    return labels


def possession_summary(labels):
    """Possessions are maximal runs of H or A; dead ball separates them. Plausibility check only."""
    lengths = [len(list(run)) for key, run in groupby(labels) if key in ("H", "A")]
    return {
        "graded_frames": int((labels != "X").sum()),
        "H": int((labels == "H").sum()),
        "A": int((labels == "A").sum()),
        "D": int((labels == "D").sum()),
        "possessions": len(lengths),
        "mean_possession_s": float(np.mean(lengths) / config.FPS) if lengths else 0.0,
    }


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="CuleCode_v1", help="output file name, without .csv")
    parser.add_argument("--out-dir", default=config.SUBMISSIONS_DIR)
    parser.add_argument("--video-ball", nargs="*", default=None,
                        help="video detection CSVs to fill gaps in the tracking ball")
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS),
                        help="decoder settings chosen on other matches")
    args = parser.parse_args()

    labels = build_labels(video_ball=args.video_ball, params=PRESETS[args.preset])
    summary = possession_summary(labels)
    print("\nlabels over graded frames: H {H}  A {A}  D {D}".format(**summary))
    print("possessions: {possessions}  mean {mean_possession_s:.1f} s".format(**summary))

    path = write_csv(os.path.join(args.out_dir, args.name + ".csv"), labels)
    print("wrote {}".format(path))
