"""Smart Tagging intervals as per-frame evidence (DATA.md, "Smart Tagging").

Smart Tagging is automatic, not ground truth. It is used only as a prior:
- SET PIECES intervals are restarts, so the ball is dead (evidence for D).
- Phases with a Team tag say which side is performing the action. Offensive phases suggest that
  team has the ball; defensive phases suggest the other side has it. Throw-ins are tagged N/A and
  are ignored here.

Time base: <start> and <end> are seconds from the first frame of the tactical video.
"""

import xml.etree.ElementTree as ET

import numpy as np

from src.atd import AWAY, HOME

OFFENSIVE = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
             "BALL IN THE BOX", "PLAYERS IN THE BOX", "LONG BALL", "ATTACKING TRANSITION"}
DEFENSIVE = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"}


def _side(team):
    """Map a Team tag to HOME/AWAY, or None for N/A and anything unrecognised."""
    name = (team or "").strip().lower()
    if name == "barcelona":
        return HOME
    if name.startswith("atletic"):
        return AWAY
    return None


def load(path, fps):
    """List of dicts with code, start_frame, end_frame (exclusive), side, in file order."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        out.append({
            "code": (instance.findtext("code") or "").strip().upper(),
            "start_frame": int(round(start_s * fps)),
            "end_frame": int(round(end_s * fps)),
            "side": _side(tags.get("Team")),
            "type": tags.get("Type"),
        })
    return out


def frame_evidence(instances, n_frames):
    """(restart, has_home, has_away) boolean arrays over the whole video."""
    restart = np.zeros(n_frames, dtype=bool)
    has_home = np.zeros(n_frames, dtype=bool)
    has_away = np.zeros(n_frames, dtype=bool)
    for event in instances:
        lo = max(0, event["start_frame"])
        hi = min(n_frames, event["end_frame"])
        if hi <= lo:
            continue
        if event["code"] == "SET PIECES":
            restart[lo:hi] = True
        side = event["side"]
        if side is None:
            continue
        if event["code"] in OFFENSIVE:
            target = has_home if side == HOME else has_away
            target[lo:hi] = True
        elif event["code"] in DEFENSIVE:
            target = has_away if side == HOME else has_home
            target[lo:hi] = True
    return restart, has_home, has_away
