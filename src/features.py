"""Per-frame evidence from the tracking: which side is closer to the ball, and is the ball out.

Distances are in metres (normalised coordinates scaled by the pitch size). Frames with no ball, or
where either team has no visible player, carry no control evidence (NaN).
"""

import numpy as np

from src.atd import AWAY, HOME
from src.config import PITCH_M


def control_evidence(xy, ball, slot_team, scale_m=10.0):
    """(control, out_of_play) for every frame.

    control is in [-1, 1]: positive when the nearest home player is closer to the ball than the
    nearest away player (by the metre gap, clipped to +/- scale_m), negative when away is closer,
    and NaN when there is no ball or a team has no visible player.

    out_of_play is True where the ball is visible outside the pitch rectangle (it has crossed a
    touchline or goal line, so it is dead).
    """
    pitch = np.array(PITCH_M, dtype=np.float32)
    ball_m = ball * pitch
    xy_m = xy * pitch
    gap = np.sqrt(np.sum((xy_m - ball_m[:, None, :]) ** 2, axis=2))
    gap = np.where(np.isnan(gap), np.inf, gap)

    nearest_home = gap[:, slot_team == HOME].min(axis=1)
    nearest_away = gap[:, slot_team == AWAY].min(axis=1)
    usable = np.isfinite(nearest_home) & np.isfinite(nearest_away)

    control = np.full(len(ball), np.nan, dtype=np.float32)
    control[usable] = np.clip((nearest_away[usable] - nearest_home[usable]) / scale_m, -1.0, 1.0)

    visible = np.isfinite(ball[:, 0])
    out_of_play = visible & ((ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1))
    return control, out_of_play
