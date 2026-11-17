"""Make clean DFL tracking look like Metrica's automatic tracking.

The training games come from a corrected optical system: the ball is always there, every player is
always there, positions are smooth. The test game's automatic tracking loses the ball on ~41% of
graded frames and drops players. A model trained on clean inputs leans on signals that will not
exist at test time, so we degrade the training inputs to match.

Everything in `measure` is an aggregate statistic of the test game's tracking -- no labels, no
human input, nothing tied to particular frames -- which is what RULES.md permits ("learning
without human labels"). It is computed once and saved to models/noise_stats.json.

Ball missingness is modelled as a two-state Markov chain (seen / missing) whose rates depend on
how fast the players are moving: the ball is lost far more often in some phases than in others,
and collective speed is the context we can measure on both datasets without labels.
"""

import json

import numba
import numpy as np

from .data import FPS

SPEED_BINS = np.array([0.0, 0.75, 1.5, 2.25, 3.0, 4.0, np.inf])
SPEED_WINDOW = 10 * FPS


def _speed(pos):
    """(T, S) speed in m/s from positions by a 2-frame centred difference, clipped at 12."""
    vel = np.full(pos.shape, np.nan, np.float32)
    vel[2:-2] = (pos[4:] - pos[:-4]) * (FPS / 4.0)
    return np.minimum(np.linalg.norm(vel, axis=-1), 12.0)


def mean_player_speed(match, window=SPEED_WINDOW):
    """(T,) mean speed of all tracked players, smoothed over one second."""
    players = np.concatenate([match.home, match.away], axis=1)
    out = np.full(match.n_frames, np.nan, np.float32)
    for start, stop in match.segments:
        with np.errstate(all="ignore"):
            spd = _speed(players[start:stop])
            m = np.nanmean(spd, axis=1)
            m = np.where(np.isnan(m), np.nanmedian(m), m)
        k = np.ones(window) / window
        out[start:stop] = np.convolve(m, k, mode="same")
    return out


def _runs(mask):
    """[(start, length)] of True runs in a boolean array."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return starts, stops - starts


def _second_diff_scale(pos):
    """Robust std of the second difference of positions -- a jitter estimate."""
    dd = pos[2:] - 2 * pos[1:-1] + pos[:-2]
    dd = dd[np.isfinite(dd)]
    return float(1.4826 * np.median(np.abs(dd))) if len(dd) else 0.0


QUANTILES = np.linspace(0, 1, 101)


def measure(match):
    """Aggregate noise statistics of one match's tracking."""
    speed = mean_player_speed(match)
    bins = np.clip(np.digitize(speed, SPEED_BINS) - 1, 0, len(SPEED_BINS) - 2)
    stats = {"speed_bins": SPEED_BINS[:-1].tolist(), "ball_missing_frac": [],
             "ball_missing_run": [], "missing_run_q": [], "seen_run_q": []}
    missing_all, run_starts, run_lengths, bin_all = [], [], [], []
    seen_starts, seen_lengths = [], []
    p_jit, b_jit, counts, spikes, seen = [], [], [], 0, 0
    for start, stop in match.segments:
        miss = np.isnan(match.ball[start:stop, 0])
        missing_all.append(miss)
        bin_all.append(bins[start:stop])
        s, l = _runs(miss)
        run_starts.append(bins[start:stop][s])
        run_lengths.append(l)
        s, l = _runs(~miss)
        seen_starts.append(bins[start:stop][s])
        seen_lengths.append(l)
        p_jit.append(_second_diff_scale(match.home[start:stop]))
        b_jit.append(_second_diff_scale(match.ball[start:stop]))
        for side in (match.home, match.away):
            counts.append(np.isfinite(side[start:stop, :, 0]).sum(axis=1))
        b = match.ball[start:stop]
        jump_prev = np.linalg.norm(b[1:-1] - b[:-2], axis=1)
        jump_next = np.linalg.norm(b[2:] - b[1:-1], axis=1)
        bridge = np.linalg.norm(b[2:] - b[:-2], axis=1)
        ok = np.isfinite(jump_prev) & np.isfinite(jump_next)
        spikes += int(np.sum(ok & (jump_prev > 2.5) & (jump_next > 2.5) & (bridge < 2.5)))
        seen += int(np.sum(np.isfinite(b[:, 0])))
    # Per-bin Markov transition rates, the maximum-likelihood estimate of a chain whose rates
    # depend on the current collective-speed bin.
    lose_n = np.zeros(len(SPEED_BINS) - 1)
    lose_d = np.zeros(len(SPEED_BINS) - 1)
    rec_n = np.zeros(len(SPEED_BINS) - 1)
    rec_d = np.zeros(len(SPEED_BINS) - 1)
    for miss, b in zip(missing_all, bin_all):
        prev, nxt, bb = miss[:-1], miss[1:], b[:-1]
        np.add.at(lose_d, bb[~prev], 1)
        np.add.at(lose_n, bb[~prev & nxt], 1)
        np.add.at(rec_d, bb[prev], 1)
        np.add.at(rec_n, bb[prev & ~nxt], 1)
    stats["p_lose"] = (lose_n / np.maximum(lose_d, 1)).tolist()
    stats["p_recover"] = (rec_n / np.maximum(rec_d, 1)).tolist()
    missing_all = np.concatenate(missing_all)
    bin_all = np.concatenate(bin_all)
    run_starts = np.concatenate(run_starts)
    run_lengths = np.concatenate(run_lengths)
    seen_starts = np.concatenate(seen_starts)
    seen_lengths = np.concatenate(seen_lengths)
    for k in range(len(SPEED_BINS) - 1):
        in_bin = bin_all == k
        stats["ball_missing_frac"].append(float(missing_all[in_bin].mean()) if in_bin.any()
                                          else float(missing_all.mean()))
        r = run_lengths[run_starts == k]
        stats["ball_missing_run"].append(float(r.mean()) if len(r) else 0.0)
        stats["missing_run_q"].append(np.quantile(r, QUANTILES).tolist() if len(r) > 10
                                      else [])
        r = seen_lengths[seen_starts == k]
        stats["seen_run_q"].append(np.quantile(r, QUANTILES).tolist() if len(r) > 10 else [])
    counts = np.concatenate(counts)
    stats.update({
        "ball_seen_frac": float(1 - missing_all.mean()),
        "player_jitter": float(np.median(p_jit)),
        "ball_jitter": float(np.median(b_jit)),
        "players_per_team_mean": float(counts.mean()),
        "ball_spike_rate": spikes / max(seen, 1),
    })
    return stats


def save_stats(stats, path):
    with open(path, "w") as handle:
        json.dump(stats, handle, indent=2)


def load_stats(path):
    with open(path) as handle:
        return json.load(handle)


@numba.njit(cache=True)
def _chain(p_on, p_off, u):
    out = np.empty(len(p_on), np.bool_)
    state = False
    for t in range(len(p_on)):
        if state:
            state = u[t] >= p_off[t]
        else:
            state = u[t] < p_on[t]
        out[t] = state
    return out


def _markov_mask(p_on, p_off, rng):
    """Boolean chain: P(off->on)=p_on[t], P(on->off)=p_off[t]. True where 'on'."""
    return _chain(p_on.astype(np.float64), p_off.astype(np.float64), rng.random(len(p_on)))


def _fill_quantiles(per_bin):
    """Quantile tables per bin; a bin without enough runs borrows its nearest neighbour's."""
    filled = [np.array(q) if len(q) else None for q in per_bin]
    for k in range(len(filled)):
        if filled[k] is None:
            near = min((j for j in range(len(filled)) if filled[j] is not None),
                       key=lambda j: abs(j - k))
            filled[k] = filled[near]
    return filled


DEAD_WANDER_SIGMA = 1.4      # m/s, stationary spread of the simulated dead-ball velocity
DEAD_WANDER_TAU = 1.0        # seconds, its time constant


def unfreeze_dead_ball(ball, labels, segments, rng):
    """Replace DFL's frozen dead-ball position with a plausible moving ball.

    The DFL feed holds the ball at one fixed position for every dead-ball frame, so "ball exactly
    still" is a perfect dead-ball detector in the training data that never exists in automatic
    tracking (there the ball is fetched, carried and placed). Each dead span becomes a smooth
    drift from where play stopped to where it restarts, plus a smooth wander whose velocity is an
    Ornstein-Uhlenbeck process. Its scale is set so that ball speed in slow phases roughly matches
    the test game's tracking (an aggregate statistic).
    """
    from .data import D
    out = ball.copy()
    dt = 1.0 / FPS
    decay = np.exp(-dt / DEAD_WANDER_TAU)
    kick = DEAD_WANDER_SIGMA * np.sqrt(1 - decay ** 2)
    for start, stop in segments:
        dead = labels[start:stop] == D
        s, lengths = _runs(dead)
        for a, n in zip(s + start, lengths):
            b = a + n
            p0 = ball[a]
            p1 = ball[b] if b < stop and np.all(np.isfinite(ball[b])) else p0
            frac = np.linspace(0, 1, n)[:, None]
            frac = frac * frac * (3 - 2 * frac)
            path = p0 + (p1 - p0) * frac
            velocity = np.zeros((n, 2))
            noise_draws = rng.normal(0, kick, (n, 2))
            for t in range(1, n):
                velocity[t] = velocity[t - 1] * decay + noise_draws[t]
            wander = np.cumsum(velocity, axis=0) * dt
            # Pin the wander to zero at both ends so the ball still leaves and re-enters play
            # where the feed says it does.
            wander -= np.linspace(0, 1, n)[:, None] * wander[-1]
            out[a:b] = (path + wander).astype(np.float32)
    return out


def apply(match, stats, clean_stats, rng):
    """A degraded copy of a clean match, matched to the test game's noise statistics."""
    from .data import Match

    home = match.home.copy()
    away = match.away.copy()
    ball = unfreeze_dead_ball(match.ball, match.labels, match.segments, rng)

    # Positional jitter: second differences of iid noise sigma have variance 6 sigma^2.
    for arr, key in ((home, "player_jitter"), (away, "player_jitter"), (ball, "ball_jitter")):
        extra = max(stats[key] ** 2 - clean_stats[key] ** 2, 0.0)
        sigma = np.sqrt(extra / 6.0)
        if sigma > 0:
            arr += rng.normal(0, sigma, arr.shape).astype(np.float32)

    # Two-state Markov chain (seen / missing) whose per-frame rates depend on the current
    # collective-speed bin, with the rates estimated on the test game's tracking.
    speed = mean_player_speed(match)
    bins = np.clip(np.digitize(speed, SPEED_BINS) - 1, 0, len(SPEED_BINS) - 2)
    p_lose = np.array(stats["p_lose"])[bins]
    p_recover = np.array(stats["p_recover"])[bins]
    for start, stop in match.segments:
        miss = _markov_mask(p_lose[start:stop], p_recover[start:stop], rng)
        ball[start:stop][miss] = np.nan

    # Occasional spurious ball detections: a one-frame jump to a nearby wrong position.
    seen = np.flatnonzero(np.isfinite(ball[:, 0]))
    n_spike = int(stats["ball_spike_rate"] * len(seen))
    if n_spike:
        idx = rng.choice(seen, n_spike, replace=False)
        angle = rng.uniform(0, 2 * np.pi, n_spike)
        radius = rng.uniform(3, 25, n_spike)
        ball[idx, 0] += (radius * np.cos(angle)).astype(np.float32)
        ball[idx, 1] += (radius * np.sin(angle)).astype(np.float32)

    # Player dropout: each player independently goes missing in runs of ~2 s so that the mean
    # number tracked per team matches the test game.
    target_missing = np.clip(1 - stats["players_per_team_mean"] / 11.0, 0, 0.5)
    if target_missing > 0:
        run_len = 2.0 * FPS
        p_rec = 1.0 / run_len
        p_drop = p_rec * target_missing / (1 - target_missing)
        for side in (home, away):
            present = np.isfinite(side[:, :, 0])
            drop = rng.random(side.shape[:2]) < p_drop
            # A cheap Markov approximation: each drop event removes a geometric-length run.
            ts, ss = np.nonzero(drop & present)
            lengths = rng.geometric(p_rec, len(ts))
            for t, s, l in zip(ts, ss, lengths):
                side[t:t + l, s] = np.nan

    return Match(name=match.name + "_noisy", home=home, away=away, ball=ball,
                 labels=match.labels, segments=match.segments, pitch=match.pitch)
