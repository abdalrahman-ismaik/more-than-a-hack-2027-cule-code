# CuléCode — Technical document (draft)

More Than a Hack 2027, Phase 1: Ball possession. Team CuléCode, Khalifa University.

> Status: draft, 2026-11-15. Numbers are from the runs recorded in the README. Sections marked **TODO** must be completed before submission.

## 1. Approach

For every frame of the tactical video we output `H` (home in possession, ball in play), `A` (away), `D` (dead ball), or `X` outside the graded halves.

The pipeline is deterministic and has no learned component:

1. **Tracking.** Read the automatic tracking (ATD). Each frame has up to 46 track slots, with positions in normalised pitch coordinates. Team membership comes from the metadata. Positions are scaled to metres (105.3 × 68.0 m).
2. **Control evidence.** For each frame with a ball, take the distance from the ball to the nearest home player and to the nearest away player. Their difference, clipped to ±10 m and scaled to [-1, 1], is the control evidence. Positive favours home. If the ball is out of the pitch rectangle, the frame is dead.
3. **Smart Tagging.** SET PIECES intervals push the decoder towards `D`. Offensive phases with a team tag nudge that team towards possession. Defensive phases nudge the other team.
4. **Video ball (optional, candidate v1.2).** A pretrained YOLO11n ball detector runs on the tactical video. Its best detection on the pitch (within a 2% margin) is mapped to pitch coordinates with the inverse homography. It is used only where the tracking ball is missing.
5. **Decoder.** A three-state Viterbi decoder (H, A, D) runs over each graded half. Transitions are sticky: the per-frame probability of a team switch is 0.002, of a stoppage 0.004, and of leaving a stoppage 0.004 (with the stoppage state itself kept with probability 0.992). Emissions are the weights in section 4.
6. **Output.** One label per frame, written as `frame,label`, validated with `validate_submission.py`.

Why this approach: the ball is missing on 41% of graded frames, so the decoder must carry possession through gaps and not guess per frame. A sticky sequence model is the simplest way to do that, and it is what the metric rewards (spellF1 counts possessions, not frames).

## 2. Assets used and dropped

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking, metadata) | Yes | Control evidence, team membership |
| Tactical Smart Tagging | Yes | Restarts and team phases (as priors only) |
| Tactical homography | Yes | Video detections mapped to the pitch |
| Tactical video (4.98 GB) | Yes | Ball detection (candidate v1.2) |
| Broadcast ATD and homography | No | Not frame-aligned to the tactical video, and not needed for the core method |
| Broadcast video (13.3 GB) | No | Same reason |
| Smart Tagging `N/A` throw-ins | Ignored | No team nudge yet |
| Event data (any source) for this match | No | Not available, and not allowed |
| Public 5-minute ground truth | Checks only | Frame alignment, format, and the pipeline check. Never used to choose weights |
| Private full-match ground truth | No | Not available |

## 3. Training data and external models

We do **not** train a model in v1 or v1.2. Every parameter is a documented constant (section 4) chosen by reasoning or on other matches. The external assets are:

| Asset | Source | Labels | Licence | Used for |
|---|---|---|---|---|
| YOLO11n football ball detector (`martinjolif/yolo-football-ball-detection`) | Hugging Face, fine-tuned on `martinjolif/football-ball-detection` | Human-labelled ball boxes on football images (dataset provided by its authors; labelling process is theirs to document) | AGPL-3.0 | Candidate v1.2 only |
| Metrica Sample Game 1 and 2 (`metrica-sports/sample-data`) | Metrica Sports, public sample data | Possession proxy derived from the event log (our derivation, see section 4) | Check the `LICENSE` file before submission | Choosing decoder weights (other matches) |

**TODO:** record the licence of the Metrica sample data, and confirm with the organisers that the AGPL-3.0 detector can be used and described in a public repository.

None of the training or tuning data comes from the provided match.

## 4. Possession rules and parameters

Emissions (log scores for H, A, D):

- `H`: `W_CONTROL × control + W_TAG × (home phase tag)`; current `W_CONTROL = 0.5`, `W_TAG = 0.5`.
- `A`: the mirror of `H` for control, plus the away tag.
- `D`: `W_OUT × (ball out of pitch) + W_RESTART × (SET PIECES) − BIAS_DEAD`; current `W_OUT = 3.0`, `W_RESTART = 3.0`, `BIAS_DEAD = 0.0`.

When the ball is missing and there is no restart, `H` and `A` tie, so the decoder keeps the previous team. This is the behaviour the metric rewards.

Forced rules:

- Frames 110,808 to 113,741 (second-half kickoff wait) are `D`.
- Each graded half is decoded separately. Everything else is `X`.

**Weight selection.** `src/sweep_metrica.py` searched 243 settings on Metrica Sample Game 1 and checked the best on Game 2. The scores are event-derived proxy labels, so they compare settings with each other and are not a score we report. The top settings are within 0.005 spellF1 of each other, so the choice is not sensitive. The current defaults give, on Game 1 and Game 2 respectively, spellF1 0.406 and 0.416, and macroF1 0.571 and 0.568.

**Possession rule choices** (where the line is drawn):

- Loose balls and occlusions: possession stays with the last team (section 1, step 5).
- Restarts: dead until the restart, and the restarting team is favoured afterwards through the tags.
- Throw-ins tagged `N/A`: no team nudge. **TODO:** decide whether to infer the team from the nearest player.
- Aerial duels and challenges: not treated as possession changes on their own.

## 5. Results

| Version | Possessions (full match) | Mean length | Notes |
|---|---|---|---|
| v1 | 274 | 16.1 s | Earlier defaults |
| v1.1 (current defaults) | 186 | 22.9 s | Weights chosen on other matches |
| v1.2 candidate (video fills tracking gaps) | 251 | 16.8 s | Video detections where tracking is missing |

Public 5-minute window (a pipeline check only, 17 possessions): v1 spellF1 0.3067 and v1.1 0.2857. These are not used to choose between versions.

Video detection quality (full match):
- Coverage: a pitch detection on 63.4% of graded frames. The tracking ball is missing on 40.7% of graded frames, and the video adds a ball position on 17.6% of graded frames.
- Agreement with the tracking ball where both are visible: within 6 m on 82% of frames, median 1.0 m. This is agreement, not accuracy against the truth.
- Two independent runs (HPC on an H100 and laptop on an RTX 2060) agree within 1 m on 99.4% of frames where both detect the ball.

## 6. Limitations

- **No validated score for v1.2.** There is no way to measure its effect on spellF1 before submission except the public window, which we do not use for choosing.
- **Possession count.** On the Metrica games the decoder gives 230 to 250 possessions, against about 300 in the event-derived proxy. The proxy may overcount, so this is not settled.
- **Ball detection noise.** The detector fires on players' feet and the crowd. The pitch filter removes most false positives, but not all.
- **Tracking is uncorrected.** Identity switches and fragmented tracks feed directly into the control evidence.
- **Smart Tagging is inferred**, so its restarts and team tags can be wrong.
- **No learned model.** The stretch goal (a PathCRF-style learned model) was not built.
- **Metrica proxy labels** are derived from events and are approximate.

**What we would fix with more time:** validate the decoder on a second source with ground truth; add ball speed and per-team control counts; infer throw-in teams; tune the transition rates against possession counts on other matches.

## 7. Reproduction

Environment:

```powershell
python -m pip install -r requirements.txt       # numpy
git clone https://github.com/metrica-sports/more-than-a-hack-2027 challenge   # validator
$env:MTAH_DATA_DIR = "<folder containing tactical/ and broadcast/>"
```

Tracking-only submission (v1.1):

```powershell
python -m src.pipeline --name CuleCode_v1_1
python challenge\evaluation\validate_submission.py submissions\CuleCode_v1_1.csv
```

Video detection (optional; about 16 minutes on one H100 NVL, measured; the RTX 2060 run took longer, time not recorded):

```powershell
python -m src.video_ball --start 11185 --stop 90652 --device 0 --out results\video_ball_h1.csv
python -m src.video_ball --start 110808 --stop 190198 --device 0 --out results\video_ball_h2.csv
```

Candidate v1.2:

```powershell
python -m src.pipeline --name CuleCode_v1_2_video --video-ball results\video_ball_h1.csv results\video_ball_h2.csv
```

Weight sweep on other matches (about 10 minutes):

```powershell
python -m src.sweep_metrica
```

The model weights for the detector are not in git. Download `yolo-football-ball-detection.pt` from the Hugging Face page above into `models/`.

**TODO:** a clean clone must regenerate the chosen submission byte for byte. Record the file hash.
