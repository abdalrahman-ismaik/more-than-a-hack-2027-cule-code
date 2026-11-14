"""Paths and constants for the tactical recording.

Every number here comes from the challenge documents (DATA.md, FAQ.md, games.py) or is a documented
model choice. None of them were fitted to the public ground truth.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "Initial Challenge")
TACTICAL_DIR = os.path.join(DATA_DIR, "tactical")
ATD_META = os.path.join(TACTICAL_DIR, "BAR-ATH_tactical_atd.xml")
ATD_RAW = os.path.join(TACTICAL_DIR, "BAR-ATH_tactical_atd.txt")
SMART_TAGGING = os.path.join(TACTICAL_DIR, "BAR-ATH_tactical_smart_tagging.xml")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")

N_FRAMES = 199319
FPS = 25
PITCH_M = (105.3, 68.0)  # length, width in metres (games.py)
N_SLOTS = 46

# Graded frames, inclusive (DATA.md, "Which frames are graded"). Outside these we write X.
GRADED_HALVES = [(11185, 90651), (110808, 190197)]
# Second half is graded from 110808 but kicks off at 113742 (FAQ.md). Those frames are dead ball.
SECOND_HALF_KICKOFF_WAIT = (110808, 113741)
