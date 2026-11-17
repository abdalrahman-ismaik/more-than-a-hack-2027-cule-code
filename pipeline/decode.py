"""From per-frame probabilities to possessions, and the training-target definition.

The ranking metric scores whole possessions, so per-frame argmax -- which flickers -- is the wrong
output. A three-state HMM (H, A, D) decoded with Viterbi gives the single most likely labelling
under a cost for every change of state. The costs are what set how many possessions come out, and
they are chosen to maximise spellF1 on held-out *training* games, never on the provided one.
"""

import numba
import numpy as np

from .data import A, D, H, X


@numba.njit(cache=True)
def _viterbi(logp, trans):
    n, k = logp.shape
    back = np.empty((n, k), np.int8)
    score = logp[0].copy()
    new = np.empty(k)
    for t in range(1, n):
        for j in range(k):
            best, arg = -1e300, 0
            for i in range(k):
                s = score[i] + trans[i, j]
                if s > best:
                    best, arg = s, i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        score[:] = new
    path = np.empty(n, np.int8)
    path[n - 1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def transition_matrix(c_switch, c_to_dead, c_from_dead):
    """Log transition costs between H, A, D. Staying put is free."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -c_switch
    t[H, D] = t[A, D] = -c_to_dead
    t[D, H] = t[D, A] = -c_from_dead
    return t


def decode(probs, segments, params, n_frames=None):
    """Viterbi inside each segment; X everywhere else."""
    n = n_frames or len(probs)
    out = np.full(n, X, np.int8)
    trans = transition_matrix(params["c_switch"], params["c_to_dead"], params["c_from_dead"])
    logp = np.log(np.clip(probs, 1e-6, 1.0)) * params.get("temp", 1.0)
    for start, stop in segments:
        out[start:stop] = _viterbi(np.ascontiguousarray(logp[start:stop]), trans)
    return out


def controlled_labels(labels, segments, min_frames=25):
    """The training target: possession only changes on *controlled* possession.

    CHALLENGE.md: "a simple touch of the ball without control", "a deflection" and "an aerial duel
    won without subsequent control" are not changes of possession. The DFL flag flips on such
    touches, so a spell of the other team shorter than `min_frames` that is followed by the
    original team again, or by a dead ball, is folded back into the original possession.
    A short spell straight after a restart is kept: after a stoppage there is always a clear
    possessor.
    """
    out = labels.copy()
    for start, stop in segments:
        seg = out[start:stop]
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(seg)]))
        runs = [[int(bounds[i]), int(bounds[i + 1]), int(seg[bounds[i]])]
                for i in range(len(bounds) - 1)]
        for i in range(1, len(runs) - 1):
            a, b, code = runs[i]
            prev, nxt = runs[i - 1][2], runs[i + 1][2]
            if code in (H, A) and prev in (H, A) and prev != code and b - a < min_frames \
                    and nxt in (prev, D):
                seg[a:b] = prev
                runs[i][2] = prev
    return out
