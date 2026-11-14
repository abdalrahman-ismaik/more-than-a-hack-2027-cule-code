"""Three-state sequence decoder: H (home in possession), A (away), D (dead ball).

Viterbi over one half at a time. The transition matrix is sticky, so the decoder prefers long
possessions and does not flicker. Emissions are hand-set, documented weights (no fitting to the
public window). Where the ball is missing and there is no restart, H and A are tied and the
decoder keeps the previous team, which is the behaviour the metric rewards.
"""

import numpy as np

H, A, D = 0, 1, 2

# Per-frame transition probabilities. Set on other matches (Metrica sample game 1, checked on game 2,
# see src/sweep_metrica.py), not on the challenge match or its public window.
P_HOME_TO_DEAD = 0.004
P_HOME_TO_AWAY = 0.002
P_DEAD_TO_TEAM = 0.004
P_DEAD_STAY = 0.992

def make_transitions(p_team_to_dead=P_HOME_TO_DEAD, p_switch=P_HOME_TO_AWAY,
                     p_dead_to_team=P_DEAD_TO_TEAM, p_dead_stay=P_DEAD_STAY):
    """3x3 per-frame transition matrix over (H, A, D). Rows sum to one."""
    stay = 1 - p_team_to_dead - p_switch
    return np.array([
        [stay, p_switch, p_team_to_dead],
        [p_switch, stay, p_team_to_dead],
        [p_dead_to_team, p_dead_to_team, p_dead_stay],
    ])


TRANSITIONS = make_transitions()

# Emission weights (log-likelihood units), set by hand.
W_CONTROL = 0.5   # how strongly the nearest-player geometry decides H vs A (other-match choice)
W_TAG = 0.5       # how strongly a Smart Tagging team phase nudges H vs A
W_OUT = 3.0       # ball visibly outside the pitch: dead
W_RESTART = 3.0   # SET PIECES interval: dead
BIAS_DEAD = 0.0   # no penalty for D in normal play (other-match choice)

FORBID = -1e4    # log-score that rules a state out completely


def emissions(control, out_of_play, restart, has_home, has_away, forced_dead=None,
              w_control=W_CONTROL, w_tag=W_TAG, w_out=W_OUT, w_restart=W_RESTART, bias=BIAS_DEAD):
    """(n, 3) log-scores for H, A, D."""
    n = len(control)
    c = np.nan_to_num(control, nan=0.0)
    ll = np.zeros((n, 3), dtype=np.float64)
    ll[:, H] = w_control * c + w_tag * has_home
    ll[:, A] = -w_control * c + w_tag * has_away
    ll[:, D] = -bias + w_out * out_of_play + w_restart * restart
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
