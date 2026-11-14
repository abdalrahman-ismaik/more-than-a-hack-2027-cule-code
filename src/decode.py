"""Three-state sequence decoder: H (home in possession), A (away), D (dead ball).

Viterbi over one half at a time. The transition matrix is sticky, so the decoder prefers long
possessions and does not flicker. Emissions are hand-set, documented weights (no fitting to the
public window). Where the ball is missing and there is no restart, H and A are tied and the
decoder keeps the previous team, which is the behaviour the metric rewards.
"""

import numpy as np

H, A, D = 0, 1, 2

# Per-frame transition probabilities. Chosen from football reasoning (possession changes every
# ~10-20 s, stoppages every ~15-40 s, dead-ball stretches of a few seconds), not from the ground truth.
P_HOME_TO_DEAD = 0.0015
P_HOME_TO_AWAY = 0.0006
P_AWAY_TO_DEAD = 0.0015
P_AWAY_TO_HOME = 0.0006
P_DEAD_TO_TEAM = 0.004
P_DEAD_STAY = 0.992

TRANSITIONS = np.array([
    [1 - P_HOME_TO_DEAD - P_HOME_TO_AWAY, P_HOME_TO_AWAY, P_HOME_TO_DEAD],
    [P_AWAY_TO_HOME, 1 - P_AWAY_TO_DEAD - P_AWAY_TO_HOME, P_AWAY_TO_DEAD],
    [P_DEAD_TO_TEAM, P_DEAD_TO_TEAM, P_DEAD_STAY],
])

# Emission weights (log-likelihood units), set by hand.
W_CONTROL = 1.0   # how strongly the nearest-player geometry decides H vs A
W_TAG = 0.5       # how strongly a Smart Tagging team phase nudges H vs A
W_OUT = 3.0       # ball visibly outside the pitch: dead
W_RESTART = 3.0   # SET PIECES interval: dead
BIAS_DEAD = 0.5   # normal play is not dead, so D starts behind by this much

FORBID = -1e4    # log-score that rules a state out completely


def emissions(control, out_of_play, restart, has_home, has_away, forced_dead=None):
    """(n, 3) log-scores for H, A, D."""
    n = len(control)
    c = np.nan_to_num(control, nan=0.0)
    ll = np.zeros((n, 3), dtype=np.float64)
    ll[:, H] = W_CONTROL * c + W_TAG * has_home
    ll[:, A] = -W_CONTROL * c + W_TAG * has_away
    ll[:, D] = -BIAS_DEAD + W_OUT * out_of_play + W_RESTART * restart
    if forced_dead is not None:
        ll[forced_dead, H] = FORBID
        ll[forced_dead, A] = FORBID
    return ll


def viterbi(ll, transitions=TRANSITIONS):
    """Most likely state path for a (n, 3) log-score array."""
    n = len(ll)
    log_t = np.log(transitions)
    delta = ll[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        scores = delta[:, None] + log_t
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + ll[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(delta.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
