"""Read the tactical tracking (ATD) into numpy arrays.

Format (DATA.md): one line per frame, "<frame_id>:<slot;slot;...>:<ball>", each group "x,y,speed"
with NaN for a missing observation. Coordinates are normalised 0-1. frame_id counts from 1.

Team membership comes from the metadata XML: the first team listed is home (FIFA/EPTS convention,
and DATA.md confirms Barcelona is home). Slots are tracklets, not players.
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Array over track slots: HOME or AWAY."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([HOME if player.get("teamId") == home_id else AWAY
                     for player in root.iter("Player")], dtype=np.int8)


def load(raw_path, n_frames, n_slots):
    """(xy, ball): xy is (n_frames, n_slots, 2) and ball is (n_frames, 2), NaN where missing."""
    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            head, players, ball_part = line.rstrip("\n").split(":")
            frame = int(head) - 1
            values = np.array(players.replace(";", ",").split(","), dtype=np.float32)
            xy[frame] = values.reshape(n_slots, 3)[:, :2]
            ball_xy = ball_part.split(",")
            ball[frame] = (float(ball_xy[0]), float(ball_xy[1]))
    return xy, ball
