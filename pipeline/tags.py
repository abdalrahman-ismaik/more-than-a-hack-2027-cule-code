"""Restart tags: Smart Tagging set pieces for the provided game, simulated ones for DFL games.

The model can only learn how far to trust a Smart Tagging set piece from training games that carry
the same kind of evidence, and the DFL games carry none. They do carry the official event log,
which records every set piece with the moment it was taken (EventTime) and the moment the referee
stopped play (DecisionTimestamp). simulate() turns those into Smart-Tagging-like intervals:

- which set pieces get a tag: per type, with the rate at which Smart Tagging tags that type in the
  provided game relative to how often the type occurs in a DFL game (counts only, no labels);
- where the interval sits relative to the restart: sampled from the provided game's tag
  boundaries measured against the restarts our own label-free model predicts (aggregate offsets,
  saved in models/possession/tag_stats.json by measure_tag_stats);
- the team: the team taking the set piece, and none for throw-ins, which Smart Tagging tags N/A.

A restart is a dict(start, stop, side, kind) in frames of the match; side is H, A or None.
"""

import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from .data import A, D, DFL_DIR, FPS, H, CACHE_DIR, asset_paths

# Smart Tagging "Type" values and DFL event elements, mapped onto one set of kinds.
SMART_KINDS = {"throw-in": "throw_in", "corner kick": "corner", "goal kick": "goal_kick",
               "free kick": "free_kick", "kick off (after goal)": "kick_off",
               "kick off (start)": "kick_off"}
DFL_KINDS = {"ThrowIn": "throw_in", "CornerKick": "corner", "GoalKick": "goal_kick",
             "FreeKick": "free_kick", "KickOff": "kick_off", "Penalty": "free_kick"}
KINDS = ("throw_in", "corner", "goal_kick", "free_kick", "kick_off")


def _side(team):
    name = (team or "").strip().lower()
    if name == "barcelona":
        return H
    if name.startswith("atletic") or name.startswith("athletic"):
        return A
    return None


def smart_restarts(game="BAR-ATH"):
    """SET PIECES instances of the provided game's Smart Tagging, in video frames."""
    out = []
    for inst in ET.parse(asset_paths(game)["smart_tagging"]).iter("instance"):
        if (inst.findtext("code") or "").strip().upper() != "SET PIECES":
            continue
        labels = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        kind = SMART_KINDS.get((labels.get("Type") or "").strip().lower())
        if kind is None:
            continue
        out.append(dict(start=int(round(float(inst.findtext("start")) * FPS)),
                        stop=int(round(float(inst.findtext("end")) * FPS)),
                        side=_side(labels.get("Team")), kind=kind))
    return sorted(out, key=lambda t: t["start"])


# -- DFL event log --------------------------------------------------------------------------

def _time(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def _section_starts(match_id):
    """{section: unix time of its first ball frame}, read once from the positions file."""
    cache = os.path.join(CACHE_DIR, "dfl_{}_t0.json".format(match_id))
    if os.path.exists(cache):
        return json.load(open(cache))
    path = glob.glob(os.path.join(DFL_DIR, "*positions_raw_observed*{}.xml".format(match_id)))[0]
    starts, section = {}, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                section = (re.search(r'GameSection="(\w+)"', line).group(1)
                           if 'TeamId="BALL"' in line else None)
            elif section and section not in starts and line.startswith("<Frame "):
                starts[section] = _time(re.search(r' T="([^"]+)"', line).group(1))
            if len(starts) == 2:
                break
    json.dump(starts, open(cache, "w"))
    return starts


def dfl_restarts(match, match_id):
    """Every set piece in a DFL game: frame it was taken, frame play was stopped, team, kind."""
    info = open(glob.glob(os.path.join(DFL_DIR, "*matchinformation*{}.xml".format(match_id)))[0],
                encoding="utf-8").read()
    home_id = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
    starts = _section_starts(match_id)
    events = open(glob.glob(os.path.join(DFL_DIR, "*events_raw*{}.xml".format(match_id)))[0],
                  encoding="utf-8").read()
    out = []
    pattern = r'<Event [^>]*EventTime="([^"]+)"[^>]*>\s*<({})\b([^>]*)>'.format(
        "|".join(DFL_KINDS))
    for m in re.finditer(pattern, events):
        when, element, attrs = _time(m.group(1)), m.group(2), m.group(3)
        team = re.search(r'Team="([^"]+)"', attrs)
        decision = re.search(r'DecisionTimestamp="([^"]+)"', attrs)
        frame = None
        for (lo, hi), section in zip(match.segments, ("firstHalf", "secondHalf")):
            f = lo + int(round((when - starts[section]) * FPS))
            if lo <= f < hi:
                frame = f
                break
        if frame is None:
            continue
        stop_frame = None
        if decision:
            stop_frame = frame - int(round((when - _time(decision.group(1))) * FPS))
        side = None
        if team:
            side = H if team.group(1) == home_id else A
        elif element == "KickOff":
            side = None
        out.append(dict(frame=frame, decision=stop_frame, side=side, kind=DFL_KINDS[element]))
    return out


def dfl_kind_counts():
    """Mean number of set pieces of each kind per DFL game."""
    from . import data
    counts = {k: 0 for k in KINDS}
    for m in data.DFL_MATCHES:
        for r in dfl_restarts(data.load_dfl(m), m):
            counts[r["kind"]] += 1
    return {k: v / len(data.DFL_MATCHES) for k, v in counts.items()}


# -- simulation ------------------------------------------------------------------------------

def simulate(restarts, stats, segments, rng):
    """Smart-Tagging-like restart intervals for a DFL game (see module docstring).

    Tags that match no predicted restart in the provided game (stats["spurious"] of them) may be
    false positives or restarts our model missed; either way the simulation adds that share of
    tags at random times, so the model does not learn to trust a tag blindly.
    """
    out = []
    starts = np.array(stats["start_offsets"])
    stops = np.array(stats["stop_offsets"])
    for r in restarts:
        if rng.random() >= stats["keep"][r["kind"]]:
            continue
        i = rng.integers(len(starts))
        lo = r["frame"] + int(starts[i])
        hi = max(lo + 1, r["frame"] + int(stops[i]))
        side = None if r["kind"] in ("throw_in", "kick_off") else r["side"]
        out.append(dict(start=lo, stop=hi, side=side, kind=r["kind"]))
    n_fake = rng.binomial(len(out), stats["spurious"]) if out else 0
    total = sum(hi - lo for lo, hi in segments)
    for _ in range(n_fake):
        pos = int(rng.integers(total))
        for lo, hi in segments:
            if pos < hi - lo:
                frame = lo + pos
                break
            pos -= hi - lo
        i = rng.integers(len(starts))
        side = [None, H, A][int(rng.integers(3))]
        out.append(dict(start=frame + int(starts[i]), stop=frame + int(stops[i]), side=side,
                        kind="spurious"))
    return sorted(out, key=lambda t: t["start"])


def tag_stats(codes, segments):
    """Everything simulate() needs, measured on the provided game without labels."""
    smart = smart_restarts()
    st = measure_tag_stats(codes, segments, smart)
    dfl = dfl_kind_counts()
    smart_n = {k: sum(t["kind"] == k for t in smart) for k in KINDS}
    st["keep"] = {k: min(1.0, smart_n[k] / dfl[k]) if dfl[k] else 0.0 for k in KINDS}
    st["spurious"] = 1.0 - st["n_matched"] / max(st["n_tags"], 1)
    st["smart_counts"] = smart_n
    st["dfl_counts_per_game"] = dfl
    return st


# -- model inputs ----------------------------------------------------------------------------

CAP_S = 60.0          # seconds: beyond this a tag says nothing about the current frame


def features(n_frames, segments, tag_list, swap=False):
    """Per-frame inputs from restart tags, team-symmetric like the rest of the features.

    t_since / t_until: seconds since the last tag started and until the next one starts (capped,
    within the half). side_last / side_next: +1 if that tag names the team the row is written
    for (home, or away when swap=True), -1 for the other team, 0 for no team.
    """
    names = ["tag_t_since", "tag_t_until", "tag_side_last", "tag_side_next", "tag_inside"]
    X = np.zeros((n_frames, len(names)), np.float32)
    X[:, 0] = X[:, 1] = CAP_S
    own, other = (A, H) if swap else (H, A)

    def sign(side):
        return 1.0 if side == own else (-1.0 if side == other else 0.0)

    frames = np.arange(n_frames)
    for lo, hi in segments:
        inside = sorted((t for t in tag_list if lo <= t["start"] < hi), key=lambda t: t["start"])
        if not inside:
            continue
        starts = np.array([t["start"] for t in inside])
        stops = np.array([t["stop"] for t in inside])
        sides = np.array([sign(t["side"]) for t in inside], np.float32)
        f = frames[lo:hi]
        last = np.searchsorted(starts, f, side="right") - 1      # last tag started at or before f
        nxt = last + 1
        has_last = last >= 0
        has_next = nxt < len(starts)
        t_since = np.where(has_last, (f - starts[np.clip(last, 0, None)]) / FPS, CAP_S)
        t_until = np.where(has_next, (starts[np.clip(nxt, None, len(starts) - 1)] - f) / FPS, CAP_S)
        X[lo:hi, 0] = np.minimum(t_since, CAP_S)
        X[lo:hi, 1] = np.minimum(t_until, CAP_S)
        X[lo:hi, 2] = np.where(has_last, sides[np.clip(last, 0, None)], 0.0)
        X[lo:hi, 3] = np.where(has_next, sides[np.clip(nxt, None, len(starts) - 1)], 0.0)
        X[lo:hi, 4] = np.where(has_last, f < stops[np.clip(last, 0, None)], False)
    return names, X


# -- quick version: rules on the probabilities --------------------------------------------------

def apply_rules(probs, tag_list, params):
    """Nudge the model's probabilities with restart tags, before decoding.

    The restart is expected `lead` frames after a tag starts. Before it (from the tag start), P(D)
    is multiplied by k_dead; after it, for `w_team` frames, P(tagged team) by k_team.
    """
    p = probs.copy()
    lead, k_dead, k_team, w_team = (params["lead"], params["k_dead"], params["k_team"],
                                    params["w_team"])
    n = len(p)
    for t in tag_list:
        e = t["start"] + lead
        lo, hi = max(0, t["start"]), min(n, e)
        if hi > lo and k_dead != 1:
            p[lo:hi, D] *= k_dead
        if t["side"] is not None and k_team != 1:
            lo, hi = max(0, e), min(n, e + w_team)
            if hi > lo:
                p[lo:hi, t["side"]] *= k_team
    p /= p.sum(axis=1, keepdims=True)
    return p


def restart_frames(codes, segments):
    """Frames where a decoded sequence goes from D to H or A (a predicted restart)."""
    out = []
    for lo, hi in segments:
        c = codes[lo:hi]
        idx = np.flatnonzero((c[1:] != D) & (c[:-1] == D)) + 1
        out.extend((idx + lo).tolist())
    return np.array(out)


def measure_tag_stats(codes, segments, smart, max_gap=15 * FPS):
    """Where Smart Tagging set-piece boundaries sit relative to predicted restarts (aggregate).

    For each tag, the nearest restart our model predicts within max_gap of the tag start; offsets
    are tag start minus restart and tag end minus restart, in frames. Only the pooled offsets and
    counts are kept.
    """
    restarts = restart_frames(codes, segments)
    start_off, stop_off = [], []
    for tag in smart:
        if len(restarts) == 0:
            break
        j = np.argmin(np.abs(restarts - tag["start"]))
        if abs(restarts[j] - tag["start"]) <= max_gap:
            start_off.append(int(tag["start"] - restarts[j]))
            stop_off.append(int(tag["stop"] - restarts[j]))
    return dict(n_tags=len(smart), n_matched=len(start_off),
                start_offsets=start_off, stop_offsets=stop_off)
