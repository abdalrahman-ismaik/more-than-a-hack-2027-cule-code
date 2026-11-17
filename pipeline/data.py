"""One in-memory format for every match, whatever the source.

A Match holds, at 25 fps:
    home, away : (T, S, 2) float32 player positions in metres, origin at the pitch centre,
                 x along the length. NaN where a slot is not observed.
    ball       : (T, 2) float32, NaN where the ball is not observed.
    labels     : (T,) int8 codes H=0 A=1 D=2 X=3 (X where there is no label).
    segments   : list of half-open [start, stop) spans of continuous play (halves). Every
                 temporal computation stays inside a segment.

Two loaders: the BAR-ATH automatic tracking (no labels) and the IDSSE / DFL open data
(labels from the per-frame BallStatus and BallPossession flags).
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, asset_paths, half_video_blocks  # noqa: E402

H, A, D, X = 0, 1, 2, 3
FPS = 25
CACHE_DIR = os.path.join(ROOT, "data", "cache")
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")
DFL_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")


@dataclass
class Match:
    name: str
    home: np.ndarray
    away: np.ndarray
    ball: np.ndarray
    labels: np.ndarray
    segments: list
    pitch: tuple = (105.0, 68.0)
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self):
        return len(self.ball)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, home=self.home, away=self.away, ball=self.ball, labels=self.labels,
                 segments=np.array(self.segments, dtype=np.int64),
                 pitch=np.array(self.pitch, dtype=np.float64), name=np.array(self.name))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(name=str(z["name"]), home=z["home"], away=z["away"], ball=z["ball"],
                   labels=z["labels"], segments=[tuple(map(int, s)) for s in z["segments"]],
                   pitch=tuple(float(v) for v in z["pitch"]))


# -- BAR-ATH automatic tracking -----------------------------------------------------------------

def load_atd(game="BAR-ATH", use_cache=True):
    """The provided ATD as a Match. Labels are all X: we never have labels for this game."""
    cache = os.path.join(CACHE_DIR, "{}_atd.npz".format(game))
    if use_cache and os.path.exists(cache):
        return Match.load(cache)

    paths = asset_paths(game)
    length, width = GAMES[game]["pitch"]
    n_frames = GAMES[game]["n_video_frames"]

    root = ET.parse(paths["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    slot_team = np.array([0 if p.get("teamId") == team_ids[0] else 1
                          for p in root.iter("Player")], dtype=np.int8)

    # "<frame_id>:<x,y,s>;...46 slots...:<x,y,s>" -> one flat numeric row per line.
    with open(paths["atd_raw"]) as handle:
        text = handle.read()
    flat = np.array(text.replace(":", ",").replace(";", ",").split(), dtype=object)
    rows = np.array([np.array(line.split(","), dtype=np.float64) for line in flat])
    del flat, text
    frame_ids = rows[:, 0].astype(np.int64)
    order = np.argsort(frame_ids)
    rows = rows[order]
    assert len(rows) == n_frames and frame_ids[order][0] == 1, "ATD is not one row per frame"

    n_slots = len(slot_team)
    players = rows[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)[:, :, :2]
    ball = rows[:, 1 + 3 * n_slots:1 + 3 * n_slots + 2]

    def to_metres(xy):
        out = np.empty_like(xy, dtype=np.float32)
        out[..., 0] = (xy[..., 0] - 0.5) * length
        out[..., 1] = (xy[..., 1] - 0.5) * width
        return out

    players = to_metres(players)
    match = Match(name=game,
                  home=players[:, slot_team == 0],
                  away=players[:, slot_team == 1],
                  ball=to_metres(ball),
                  labels=np.full(n_frames, X, dtype=np.int8),
                  segments=[(start, stop + 1) for _, start, stop in half_video_blocks(game)],
                  pitch=(length, width))
    match.save(cache)
    return match


# -- DFL / IDSSE open data -------------------------------------------------------------------

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
_BALLFLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _dfl_file(kind, match_id):
    hits = glob.glob(os.path.join(DFL_DIR, "DFL_*_{}_*{}.xml".format(kind, match_id)))
    if not hits:
        raise FileNotFoundError("no DFL {} file for {} in {}".format(kind, match_id, DFL_DIR))
    return hits[0]


def load_dfl(match_id, use_cache=True):
    """One IDSSE match as a Match, with labels D (BallStatus 0) else H/A from BallPossession."""
    cache = os.path.join(CACHE_DIR, "dfl_{}.npz".format(match_id))
    if use_cache and os.path.exists(cache):
        return Match.load(cache)

    info = open(_dfl_file("matchinformation", match_id), encoding="utf-8").read()
    home_id = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
    guest_id = re.search(r'GuestTeamId="([^"]+)"', info).group(1)
    length = float(re.search(r'PitchX="([\d.]+)"', info).group(1)) if "PitchX" in info else 105.0
    width = float(re.search(r'PitchY="([\d.]+)"', info).group(1)) if "PitchY" in info else 68.0

    # section -> {"ball": {N: (x, y, poss, status)}, "players": {(team, person): {N: (x, y)}}}
    sections = {}
    current = None
    with open(_dfl_file("positions_raw_observed", match_id), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                m = _FRAMESET.search(line)
                section, team, person = m.groups()
                store = sections.setdefault(section, {"ball": {}, "players": {}})
                if team == "BALL":
                    current = ("ball", store["ball"])
                else:
                    current = ("player", store["players"].setdefault((team, person), {}))
            elif line.startswith("<Frame "):
                m = _FRAME.search(line)
                n, x, y = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if current[0] == "ball":
                    flags = _BALLFLAGS.search(line)
                    current[1][n] = (x, y, int(flags.group(1)), int(flags.group(2)))
                else:
                    current[1][n] = (x, y)

    homes, aways, balls, labels, segments = [], [], [], [], []
    offset = 0
    for section in ("firstHalf", "secondHalf"):
        store = sections[section]
        ns = np.array(sorted(store["ball"]))
        n0, t = ns[0], ns[-1] - ns[0] + 1
        ball = np.full((t, 2), np.nan, np.float32)
        lab = np.full(t, X, np.int8)
        for n, (x, y, poss, status) in store["ball"].items():
            ball[n - n0] = (x, y)
            lab[n - n0] = D if status == 0 else (H if poss == 1 else A)
        sides = {True: [], False: []}
        for (team, _), frames in store["players"].items():
            if team not in (home_id, guest_id):
                continue  # referees and assistants
            track = np.full((t, 2), np.nan, np.float32)
            for n, xy in frames.items():
                if 0 <= n - n0 < t:
                    track[n - n0] = xy
            sides[team == home_id].append(track)
        homes.append(np.stack(sides[True], axis=1))
        aways.append(np.stack(sides[False], axis=1))
        balls.append(ball)
        labels.append(lab)
        segments.append((offset, offset + t))
        offset += t

    def pad_cat(parts):
        width_slots = max(p.shape[1] for p in parts)
        return np.concatenate([np.pad(p, ((0, 0), (0, width_slots - p.shape[1]), (0, 0)),
                                      constant_values=np.nan) for p in parts])

    match = Match(name="dfl_" + match_id, home=pad_cat(homes), away=pad_cat(aways),
                  ball=np.concatenate(balls), labels=np.concatenate(labels),
                  segments=segments, pitch=(length, width))
    match.save(cache)
    return match
