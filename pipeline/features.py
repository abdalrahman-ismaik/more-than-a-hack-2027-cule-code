"""Per-frame features from a Match, identical for DFL and for the provided ATD.

Design rules:
  * Identity-free. Slots are tracklets, not people, so everything is a per-team aggregate
    (nearest distance, counts within a radius, centroid, spread, mean speed).
  * Team-symmetric. Features are written "home vs away" and each team's geometry is expressed in
    its own attacking direction, so swapping the two teams' arrays and the H/A labels gives an
    equally valid training example. That is used for augmentation and test-time averaging.
  * Offline. The label of a frame depends on what happens next (who ends up with a loose ball),
    so window features look both backwards and forwards.
  * Missing data is explicit. The ball is interpolated through gaps, but every frame also carries
    how long ago and how far ahead the ball was last/next seen, so the model can discount it.
"""

import numpy as np

from .data import FPS

# A control event: the nearest player of a team is close to the ball, closer than any opponent,
# and moving with it. These are the CHALLENGE.md "controlled possession" proxy, not tuned values.
CTRL_DIST, CTRL_MARGIN, CTRL_RELV = 1.5, 1.0, 3.0
LOOSE_DIST = 2.5
KNOWN_GAP = 12          # frames: ball positions interpolated over shorter gaps count as known
CAP = 750               # frames: cap for time-since / time-until features


def _ffill_index(mask):
    idx = np.where(mask, np.arange(len(mask)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(mask):
    n = len(mask)
    idx = np.where(mask, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def _interp_ball(ball):
    """Linear interpolation through gaps (hold at the ends) plus gap bookkeeping."""
    n = len(ball)
    obs = np.isfinite(ball[:, 0])
    last = _ffill_index(obs)
    nxt = _bfill_index(obs)
    t_since = np.where(last >= 0, np.arange(n) - last, CAP).astype(np.float32)
    t_until = np.where(nxt < n, nxt - np.arange(n), CAP).astype(np.float32)
    out = np.full_like(ball, np.nan)
    if obs.any():
        idx = np.flatnonzero(obs)
        for k in range(2):
            out[:, k] = np.interp(np.arange(n), idx, ball[idx, k])
    return out, obs, np.minimum(t_since, CAP), np.minimum(t_until, CAP)


def _despike(ball):
    """Drop single-frame ball detections that jump away and straight back."""
    b = ball.copy()
    jp = np.linalg.norm(b[1:-1] - b[:-2], axis=1)
    jn = np.linalg.norm(b[2:] - b[1:-1], axis=1)
    br = np.linalg.norm(b[2:] - b[:-2], axis=1)
    spike = np.zeros(len(b), bool)
    with np.errstate(invalid="ignore"):
        spike[1:-1] = (jp > 2.5) & (jn > 2.5) & (br < 2.5)
    b[spike] = np.nan
    return b


def _vel(pos, half=2, cap=12.0):
    """Centred-difference velocity in m/s, magnitude capped; NaN where either end is missing.

    cap is 12 m/s for players (faster is a tracklet switch, not a sprint) and 40 m/s for the ball.
    """
    v = np.full(pos.shape, np.nan, np.float32)
    v[half:-half] = (pos[2 * half:] - pos[:-2 * half]) * (FPS / (2.0 * half))
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(sp > cap, cap / sp, 1.0)
    return v * scale


def _win(sig, w, kind):
    """NaN-aware mean of sig over a past [t-w, t], future [t, t+w] or centred window."""
    n = len(sig)
    finite = np.isfinite(sig)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, sig, 0.0))))
    cc = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(n)
    if kind == "p":
        lo, hi = np.maximum(0, i - w), i + 1
    elif kind == "f":
        lo, hi = i, np.minimum(n, i + w + 1)
    else:
        lo, hi = np.maximum(0, i - w), np.minimum(n, i + w + 1)
    cnt = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def _attack_dir(home, length):
    """+1 if the home side attacks towards +x in this segment, else -1.

    The goalkeeper is the team's deepest player almost all the time, so the end where the
    team's extreme player sits closest to the goal line is the end it defends.
    """
    x = home[:, :, 0]
    with np.errstate(all="ignore"):
        lo = np.nanmedian(np.nanmin(x, axis=1))
        hi = np.nanmedian(np.nanmax(x, axis=1))
    return 1.0 if (lo + length / 2) < (length / 2 - hi) else -1.0


def _team_block(players, vel, bpos, bvel, direction, prefix):
    """Per-team features for one segment, in the team's own attacking direction."""
    f = {}
    present = np.isfinite(players[:, :, 0])
    n = present.sum(axis=1).astype(np.float32)
    rel = players - bpos[:, None, :]
    dist = np.linalg.norm(rel, axis=-1)
    dist = np.where(present, dist, np.inf)
    order = np.argsort(dist, axis=1)
    d_sorted = np.take_along_axis(dist, order, axis=1)
    d1 = np.where(np.isfinite(d_sorted[:, 0]), d_sorted[:, 0], np.nan)
    d2 = np.where(np.isfinite(d_sorted[:, 1]), d_sorted[:, 1], np.nan)
    nearest = order[:, 0]
    t = np.arange(len(players))
    v_near = vel[t, nearest]
    relv = np.linalg.norm(v_near - bvel, axis=-1)
    spd = np.linalg.norm(vel, axis=-1)
    x_own = players[:, :, 0] * direction
    # Robust shape statistics: one stray or duplicated track must not move them.
    with np.errstate(all="ignore"):
        qx = np.nanpercentile(x_own, [10, 25, 50, 75, 90], axis=1)
        qy = np.nanpercentile(players[:, :, 1], [10, 25, 50, 75, 90], axis=1)
        cx, cy = qx[2], qy[2]
        f[prefix + "sx"] = qx[3] - qx[1]
        f[prefix + "sy"] = qy[3] - qy[1]
        f[prefix + "len"] = qx[4] - qx[0]
        f[prefix + "wid"] = qy[4] - qy[0]
        f[prefix + "spd"] = np.nanmedian(spd, axis=1)
        f[prefix + "vcx"] = np.nanmedian(vel[:, :, 0] * direction, axis=1)
    f[prefix + "d1"] = d1
    f[prefix + "d2"] = d2
    f[prefix + "n3"] = (dist < 3).sum(axis=1).astype(np.float32)
    f[prefix + "n8"] = (dist < 8).sum(axis=1).astype(np.float32)
    f[prefix + "relv"] = relv
    f[prefix + "nspd"] = np.linalg.norm(v_near, axis=-1)
    f[prefix + "cx"] = cx
    f[prefix + "dcent"] = np.hypot(bpos[:, 0] * direction - cx, bpos[:, 1] - cy)
    return f


def segment_features(home, away, ball, length, width):
    """Feature dict for one continuous segment."""
    ball = _despike(ball)
    bpos, obs, t_since, t_until = _interp_ball(ball)
    gap = np.where(obs, 0, t_since + t_until)
    known = gap <= KNOWN_GAP
    bvel = _vel(bpos, cap=40.0)
    bspd = np.linalg.norm(bvel, axis=-1)
    bacc = np.full(len(bpos), np.nan, np.float32)
    bacc[1:] = np.linalg.norm(np.diff(bvel, axis=0), axis=-1)
    d = _attack_dir(home, length)

    f = {}
    f["b_obs"] = obs.astype(np.float32)
    f["b_since"] = t_since
    f["b_until"] = t_until
    f["b_gap"] = np.minimum(gap, 2 * CAP).astype(np.float32)
    f["b_x"] = bpos[:, 0] * d
    f["b_y"] = np.abs(bpos[:, 1])
    f["b_spd"] = np.where(known, bspd, np.nan)
    f["b_acc"] = np.where(known, bacc, np.nan)
    oob = np.maximum(np.abs(bpos[:, 0]) - length / 2, np.abs(bpos[:, 1]) - width / 2)
    f["b_oob"] = np.where(known, oob, np.nan)

    hv, av = _vel(home), _vel(away)
    f.update(_team_block(home, hv, bpos, bvel, d, "h_"))
    f.update(_team_block(away, av, bpos, bvel, -d, "a_"))

    allspd = np.concatenate([np.linalg.norm(hv, axis=-1), np.linalg.norm(av, axis=-1)], axis=1)
    with np.errstate(all="ignore"):
        f["all_spd"] = np.nanmean(allspd, axis=1)
        f["all_slow"] = np.nanmean(np.where(np.isfinite(allspd), allspd < 1.0, np.nan), axis=1)

    near_diff = np.clip(f["a_d1"] - f["h_d1"], -20, 20)
    near_diff = np.where(known, near_diff, np.nan)
    f["near_diff"] = near_diff

    def ctrl(me, other, dist):
        with np.errstate(invalid="ignore"):
            return known & (f[me + "d1"] < dist) & (f[other + "d1"] > f[me + "d1"] + CTRL_MARGIN) \
                & (f[me + "relv"] < CTRL_RELV)

    ch, ca = ctrl("h_", "a_", CTRL_DIST), ctrl("a_", "h_", CTRL_DIST)
    lh, la = ctrl("h_", "a_", LOOSE_DIST), ctrl("a_", "h_", LOOSE_DIST)
    f["ctrl_h"], f["ctrl_a"] = ch.astype(np.float32), ca.astype(np.float32)
    f["loose_h"], f["loose_a"] = lh.astype(np.float32), la.astype(np.float32)

    # Who controlled the ball last, and who controls it next, and how long ago / ahead.
    n = len(bpos)
    any_ctrl = ch | ca
    team_code = np.where(ch, 1.0, np.where(ca, -1.0, 0.0))
    last = _ffill_index(any_ctrl)
    nxt = _bfill_index(any_ctrl)
    f["last_ctrl"] = np.where(last >= 0, team_code[np.maximum(last, 0)], 0.0)
    f["t_last_ctrl"] = np.where(last >= 0, np.minimum(np.arange(n) - last, CAP), CAP)
    f["next_ctrl"] = np.where(nxt < n, team_code[np.minimum(nxt, n - 1)], 0.0)
    f["t_next_ctrl"] = np.where(nxt < n, np.minimum(nxt - np.arange(n), CAP), CAP)
    # Same for each team separately: time since / until that team last / next controlled it.
    for code, name in ((ch, "h"), (ca, "a")):
        last = _ffill_index(code)
        nxt = _bfill_index(code)
        f["t_last_" + name] = np.where(last >= 0, np.minimum(np.arange(n) - last, CAP), CAP)
        f["t_next_" + name] = np.where(nxt < n, np.minimum(nxt - np.arange(n), CAP), CAP)

    # Temporal context.
    for w in (12, 25, 50, 125):
        f["near_diff_p%d" % w] = _win(near_diff, w, "p")
        f["near_diff_f%d" % w] = _win(near_diff, w, "f")
    for w in (25, 75):
        for name in ("ctrl_h", "ctrl_a", "loose_h", "loose_a"):
            f["%s_p%d" % (name, w)] = _win(f[name], w, "p")
            f["%s_f%d" % (name, w)] = _win(f[name], w, "f")
    for w in (25, 125):
        f["b_obs_p%d" % w] = _win(f["b_obs"], w, "p")
        f["b_obs_f%d" % w] = _win(f["b_obs"], w, "f")
    for w in (12, 50, 125, 250):
        f["all_spd_c%d" % w] = _win(f["all_spd"], w, "c")
    for w in (50, 125):
        f["all_spd_p%d" % w] = _win(f["all_spd"], w, "p")
        f["all_spd_f%d" % w] = _win(f["all_spd"], w, "f")
        f["all_slow_c%d" % w] = _win(f["all_slow"], w, "c")
    for w in (12, 50):
        f["b_spd_p%d" % w] = _win(f["b_spd"], w, "p")
        f["b_spd_f%d" % w] = _win(f["b_spd"], w, "f")
    with np.errstate(invalid="ignore"):
        out_flag = np.where(np.isfinite(f["b_oob"]), (f["b_oob"] > 0).astype(np.float32), np.nan)
    for w in (25, 125):
        f["b_out_p%d" % w] = _win(out_flag, w, "p")
        f["b_out_f%d" % w] = _win(out_flag, w, "f")
    for w in (25, 75):
        f["h_vcx_c%d" % w] = _win(f["h_vcx"], w, "c")
        f["a_vcx_c%d" % w] = _win(f["a_vcx"], w, "c")
    for name in ("h_sy", "a_sy", "h_sx", "a_sx", "h_cx", "a_cx"):
        f[name + "_c50"] = _win(f[name], 50, "c")
    return f


def compute(match, swap=False):
    """(names, X) for every frame of a match; rows outside its segments are NaN.

    swap=True exchanges the two teams first. Combined with relabelling H<->A that is a second,
    equally valid view of the same game.
    """
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom")
    home, away = (match.away, match.home) if swap else (match.home, match.away)
    length, width = match.pitch
    names, X = None, None
    for start, stop in match.segments:
        f = segment_features(home[start:stop], away[start:stop], match.ball[start:stop],
                             length, width)
        if names is None:
            names = list(f)
            X = np.full((match.n_frames, len(names)), np.nan, np.float32)
        X[start:stop] = np.column_stack([f[k] for k in names]).astype(np.float32)
    return names, X


def swap_labels(labels):
    out = labels.copy()
    out[labels == 0] = 1
    out[labels == 1] = 0
    return out
