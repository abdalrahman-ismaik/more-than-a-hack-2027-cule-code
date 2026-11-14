# More Than a Hack 2027

**Team workspace for the More Than a Hack football analytics hackathon at Talent Arena.**

> **Team name:** CuléCode (file names use `CuleCode`)  
> **Team lead / contact:** _Add name and preferred contact_  
> **Last updated:** 2026-11-15

Use this README as the team's shared dashboard. Update the roster, status, tasks, decisions, and file index as work progresses. The detailed event summary and setup guidance is in [MORE_THAN_A_HACK_2027.md](MORE_THAN_A_HACK_2027.md).

## Project at a glance

More Than a Hack is organized by Mobile World Capital Barcelona, Barça Innovation Hub (BIHUB), and Metrica Sports. It brings teams together to work with real FC Barcelona match data and develop football analytics ideas.

The first challenge asks teams to estimate team possession for every frame using provided automatic tracking data and homography. The top 20 teams advance to an open-ended innovation challenge; the top 5 teams are invited to pitch at Talent Arena in Barcelona.

| Item | Details |
|---|---|
| Official event page | [More Than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027) |
| Organizer contact | [hackathon.talentarena@mobileworldcapital.com](mailto:hackathon.talentarena@mobileworldcapital.com) |
| Team size | 5 members |
| University | Khalifa University, Abu Dhabi |
| Current phase | Initial Challenge — open |
| Challenge repository | [Metrica Sports: More Than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) |
| Challenge data | [Organizer-provided Google Drive folder](https://drive.google.com/drive/folders/1oiEl5sE3yIElIPluOwj5J1uYgFM604uh?usp=drive_link) |
| Submission form | Shared by email to registered participants |
| **Deadline** | **Nov 17, 2026, 11:59 PM CET** = **Nov 18, 2026, 2:59 AM Abu Dhabi time** |

## The problem, explained

**For every frame of one football match, say which team has the ball.**

- Input: one full match (FC Barcelona vs Athletic Club, LaLiga 2024-25), 199,319 frames at 25 fps (about 2 hours 15 minutes, of which 158,857 frames are graded).
- Output: one CSV with 199,319 rows, `frame,label`. Labels:
  - `H`: home team (Barcelona) has the ball, and it's in play
  - `A`: away team (Athletic) has the ball, and it's in play
  - `D`: dead ball, the ball is not in play
  - `X`: no answer (ignored outside graded frames)
- Scored automatically against a private full-match ground truth. Only the public 5 minutes are visible to us, and those are **for checking that our pipeline is correct, not for training or tuning**.

**What is ranked: `spellF1`.** It counts whole possessions. A possession is a continuous run of `H` or `A`, and each one counts once, whether it lasts 2 seconds or 60. `macroF1` (per frame) is the tie-breaker only. A system can score 97% per frame and still invent hundreds of fake possessions, so the goal is clean, long possessions with correct transitions.

**Why it's hard:**
- The ball is missing on 41% of graded frames, often during the interesting moments (crowded boxes, aerial duels).
- Tracking has 46 anonymous track slots for 22 players. Slots appear, vanish, and swap identities. There are no player names.
- Smart Tagging is automatic. It can be wrong, and `N/A` team tags include every throw-in.
- The metric rewards getting *transitions* right. Repeating the previous label through an occlusion is free; a wrong label at a handover is expensive.

**Rules we must follow:**
- Labels must come from an algorithm. No hand labels on this match, in any amount, for any purpose.
- No use of the private ground truth, and no human-produced data about this match (event data, corrected tracking).
- Hand-labelled data from **other matches** is allowed, as long as we document it.
- No cloud LLM may receive the videos, datasets, or extracts of them.
- The repo must be public, must not contain the data, and must regenerate the submission file exactly.
- The technical document must explain every method, and list the training data used.

## Data status

| File | Status |
|---|---|
| `BAR-ATH_tactical_smart_tagging.xml` | Have |
| `BAR-ATH_tactical_atd.xml` | Have |
| `BAR-ATH_tactical_homography.csv` | Have |
| `BAR-ATH_broadcast_atd.xml` | Have |
| `BAR-ATH_broadcast_homography.csv` | Have |
| `BAR-ATH_tactical_atd.txt` | Have (186 MB) |
| `BAR-ATH_broadcast_atd.txt` | Have (135 MB) |
| `BAR-ATH_tactical.mp4`, `BAR-ATH_broadcast.mp4` | Not in the folder yet. Not needed for the core plan. |

The files are in `Initial Challenge/` (not `data/`). Point the code at that folder with `MTAH_DATA_DIR`:

```powershell
$env:MTAH_DATA_DIR = "C:\Users\4bais\fcb-hack-2027\Initial Challenge"
```

The data directory is git-ignored.

### Insurance submission (done 2026-11-15)

- Baseline run: `challenge/starter/baseline.py`, ball on 56.9% of frames, matching DATA.md.
- Validated: `submissions/CuleCode_insurance_smooth50.csv`, 199,319 rows, 669 possessions.
- Public 5-minute window: spellF1 0.3256, macroF1 0.6207 (the same numbers as the official board).
- Full-match spellF1 is 0.2433 (the published floor). This file is the fallback.

## Repo and how to run it (v1 pipeline)

The team repo is this folder. Our code lives in `src/`. The organisers' repository is kept in `challenge/` as its own clone, so each member clones it separately, and it is git-ignored here.

```powershell
# one-time
python -m pip install -r requirements.txt
git clone https://github.com/metrica-sports/more-than-a-hack-2027 challenge

# every run
$env:MTAH_DATA_DIR = "C:\Users\4bais\fcb-hack-2027\Initial Challenge"   # or the server path
python -m src.pipeline --name CuleCode_v1                                  # writes submissions/CuleCode_v1.csv
python challenge\evaluation\validate_submission.py submissions\CuleCode_v1.csv
```

| File | What it does |
|---|---|
| `src/config.py` | Paths and constants, each with its source in the challenge docs |
| `src/atd.py` | Reads the tracking file and the team membership |
| `src/tags.py` | Smart Tagging intervals as evidence: restarts (dead ball) and team phases |
| `src/features.py` | Control evidence (nearest home vs away player, in metres) and ball out of play |
| `src/decode.py` | 3-state Viterbi (H, A, D) with sticky transitions and documented weights |
| `src/pipeline.py` | End to end: writes the `frame,label` CSV and prints possession counts |

### v1 status (2026-11-15)

- Full match runs in about a minute and gives 274 possessions, mean 16.1 s, with `X` only outside the graded halves.
- Passes the official validator.
- Public 5-minute window: macroF1 0.6338, spellF1 0.3067. That window has 17 possessions, so this is a pipeline check only, not a ranking.
- Next: validate on the Metrica sample matches (other games, allowed) to choose the weights, not on the public window.

### HPC (KU H100 teaching server)

- Access: PuTTY, host `10.11.45.11`, port 22, SSH. Use KU VPN off campus. Use only your own credentials.
- Shared environment: `conda activate /data/shared_envs/gpu_pytorch`. Use it in job scripts, not your own installs.
- Limits for students: `gpu-debug` (1 GPU, up to 1 hour) for short tests, `gpu-student` (1 to 2 GPUs, up to 6 hours) for project jobs. No `gpu-long`.
- Run jobs with `sbatch` (or `srun` for short interactive tests). Never run heavy work in the login shell.
- Template: `hpc/run_pipeline.sbatch`. Fill in the CPU partition name (ask the admins). Output goes to `logs/`.
- Before uploading match data to the server, confirm with the organisers that this is permitted under the data terms.

## Proposed plan

The core idea: **a possession is a stable state of the 22 players and the ball, and the ball is often missing.** So we infer possession from *where the players are* as well as the ball, and we decode a whole sequence with a model that prefers long, consistent possessions.

### Pipeline

```
ATD (tracking) ──► per-frame features ──► 3-state decoder (H / A / D) ──► labels ──► validate ──► submission
                    ▲                         ▲
Smart Tagging ──────┘ restart + team priors    │
Homography (valid flag, quality)               │
```

**Stage 1 — Per-frame features (`src/features.py`)**
- Ball visible, ball speed, distance from ball to each team's nearest player
- Number of players per team within a control radius (about 3 m, in metres using 105.3 × 68.0)
- Team in control: which team has more players close to the ball, with a margin
- Restart signal from Smart Tagging (Throw-in, Corner, Goal Kick, Free Kick, Kick Off)
- Homography `valid` flag, to down-weight unreliable frames

**Stage 2 — Sequence decoder (`src/decode.py`)**
- Three hidden states: H, A, D. Viterbi (or forward-backward HMM) over the whole half.
- Sticky transition matrix, so possessions stay long. Transitions into D are cheaper at restarts and when the ball stays dead for a long time.
- Emission from the team-in-control evidence. Dead-ball emission from missing ball plus restart evidence.
- Output is one label per frame, so the decoder never needs `X`.

**Stage 3 — Possession rules (`src/rules.py`)**
- Keep the last team through loose-ball and occluded periods.
- After a restart, give the team from the Smart Tagging team tag. If the tag is `N/A` (throw-ins), infer from which side is closest to the ball.
- Force `D` from a restart until the ball moves again.
- Kickoff-wait frames in the second half (110,808 to 113,741) are `D`.

**Stage 4 — Validation and submission (`src/make_submission.py`)**
- `evaluation/validate_submission.py` on the exact file we upload.
- Check possession counts look plausible, not thousands of tiny possessions.
- Public 5 minutes: alignment and format only.
- Metrica Sample Game 1 and 2 (other matches, see below): sanity-check the method on a different match.

### Stretch goal (only if Stage 1 to 4 are working by Nov 16 evening)

A learned possession model in the style of **PathCRF** (KDD 2026): a set-transformer over player trajectories plus a CRF for consistent state sequences. It's ball-free, which is exactly our problem. Trained on external data about other games (Sportec Open DFL, or Metrica sample), not on this match. Only attempt this if the decoder is stable. We don't have time for a second pipeline that isn't ready by the deadline.

### Validation without cheating

- Use the public 5 minutes to check that frames line up, that labels mean what we think, and that the file is well formed. Never choose between versions based on them.
- Use Metrica Sample Game 1 and 2 (other matches, with ball tracking and event data) to sanity-check the method on a different match. We write down that these are other-match data, and that their labels are event-derived (so they are human-produced data about *other* games, which is allowed).
- For parameter choices, use football reasoning and the metric definitions, not the public window.

## Reused code and data

We don't start from zero. These repos are cloned into `external/` (git-ignored, do not commit their data):

| Repo | What we take | Licence | Notes |
|---|---|---|---|
| [`metrica-sports/more-than-a-hack-2027`](https://github.com/metrica-sports/more-than-a-hack-2027) (in `challenge/`) | Loaders, scorer, baselines, validator | MIT (code) | Official. The scorer decides the competition. |
| [`hyunsungkim-ds/pathcrf`](https://github.com/hyunsungkim-ds/pathcrf) (in `external/pathcrf`) | Possession-from-trajectories idea, `applications/team_poss.py` episode logic, CRF smoothing design | MPL-2.0 | KDD 2026, ball-free. Pretrained models are for Sportec format and need an adapter. |
| [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data) (in `external/metrica-sample-data`) | Two other matches with ball tracking and events, for validation | Check `LICENSE` terms before use | Other-match data, so allowed. Document it. |
| [`PySport/kloppy`](https://github.com/PySport/kloppy) (pip) | Loaders for Metrica and Sportec formats | BSD-3 | Reads tracking and event data. |
| [`lRomul/ball-action-spotting`](https://github.com/lRomul/ball-action-spotting) | Reference only: 1st place SoccerNet 2023, video-based | MIT | Video ball detection. Not needed for v1. |

Literature used for design: tracking-data possession and HMM-based state segmentation are well established; PathCRF (arXiv 2602.12080) is the closest recent work to our exact setting.

## Team and tasks

Five members. Each person owns one area, with one clear deliverable and a completion condition. Everyone reviews one other person's pull request.

| ID | Member | Area | Deliverables | Done when |
|---|---|---|---|---|
| **A** | Abd Alrahman Ismaik | Data, baseline and validation | Load ATD and homography, data QA (missing ball, frame alignment, track count), run `starter/baseline.py`, validation script, first insurance submission | Insurance submission validates and is uploaded to the team folder |
| **B** | Faisal Alzarooni | Core decoder | `src/decode.py`: 3-state Viterbi / HMM with sticky transitions, emission model | Decoder runs end to end on the full match, output validates |
| **C** | Mohamed Elashmouney | Features and statistics | `src/features.py`: control-radius features, team-in-control evidence, dead-ball evidence; statistical checks on possession counts and durations | Feature table written to disk, checked against the metric definitions |
| **D** | Hardik Sharma | Smart Tagging, restarts and reused code | `src/rules.py`: restart logic, team priors from tags, `N/A` throw-in handling; adapt PathCRF ideas and the Metrica sample loader; stretch learned model only if B is stable | Rules implemented, unit-tested on Metrica sample matches |
| **E** | Osama Qa'adan | Repo, reproducibility and document | Public repo with `.gitignore`, pinned `requirements.txt`, one command to regenerate, technical document (approach, assets used and dropped, training data, possession rules, limitations, reproduction) | Clean clone reproduces the file byte for byte; document reviewed |

### Coordination

- Daily check-in at a fixed time (set in the communication channel). 15 minutes, each person: done, next, blocked.
- Shared channel: _add link_. Data and the team folder: organisers' rules apply.
- Code goes through pull requests, one reviewer per PR.
- Tasks are tracked in the table below. Update status at each check-in.

## Timeline

| When | Milestone | Owner |
|---|---|---|
| **Nov 15 (today)** | Data complete in `data/`; baseline runs; insurance submission validated | A (lead), E (repo created) |
| **Nov 16, midday** | Features (C) and decoder (B) run on the full match; rules (D) plugged in | B, C, D |
| **Nov 16, evening** | First full submission generated and validated; stretch decision made | Everyone |
| **Nov 17, morning** | Freeze the method. Regenerate from clean clone, validate, check counts | E (lead), A |
| **Nov 17, by 14:00** | Make repo public, finish document, **submit** | E, everyone reviews |
| **Nov 17, 23:59 CET** | Hard deadline (do not use this as the submission time) | — |

## Task tracker

| ID | Task | Owner | Priority | Status | Due | Notes |
|---|---|---|---|---|---|---|
| T-01 | Confirm roster and team registration | Team lead | High | Done | Nov 10 | Osama accepted (Nov 15) |
| T-02 | Get `BAR-ATH_tactical_atd.txt` and the rest of the data | A | High | Done | Nov 15 | In `Initial Challenge/` |
| T-03 | Run baseline and validation; insurance submission | A | High | Done | Nov 15 | `submissions/CuleCode_insurance_smooth50.csv` |
| T-04 | Set up repo, `.gitignore`, pinned requirements | E | High | Not started | Nov 15 | Keep private until the end |
| T-05 | Clone and inspect PathCRF, Metrica sample data | D | Medium | Done | Nov 15 | Cloned to `external/` |
| T-06 | Features module (control radius, team evidence) | C | High | Done (v1) | Nov 16 | `src/features.py`; statistics checks still to do |
| T-07 | Viterbi decoder with sticky transitions | B | High | Done (v1) | Nov 16 | `src/decode.py`; weights to be checked on other matches |
| T-08 | Restart and tag rules | D | High | Done (v1) | Nov 16 | `src/tags.py`; throw-in and `N/A` handling still open |
| T-09 | Full-match submission generated and validated | B + C + D | High | Done (v1) | Nov 16 | `submissions/CuleCode_v1.csv` |
| T-13 | Metrica sample matches as a validation set (other games) | C + D | High | Not started | Nov 16 | Choose weights here, not on the public window |
| T-14 | Git repo: first commit, add members, push | E | High | In progress | Nov 15 | Local repo created; remote not yet set |
| T-15 | HPC access checked, CPU partition name confirmed | A | Medium | Not started | Nov 16 | `hpc/run_pipeline.sbatch` template ready |
| T-10 | Technical document drafted | E | High | Not started | Nov 17 | Required deliverable |
| T-11 | Final clean-clone reproduction check | E | High | Not started | Nov 17 | Must produce the same file |
| T-12 | Submit | Team lead | High | Not started | Nov 17 | Target 14:00 |

## Team decisions and notes

| Date | Decision / update | Owner | Follow-up |
|---|---|---|---|
| 2026-11-10 | Repository dashboard created. | — | — |
| 2026-11-10 | Organizer launch email: Initial Challenge open. Deadline Nov 17, 11:59 PM CET. | — | Contact organizers about late registration. |
| 2026-11-10 | Cloned the official challenge repository into `challenge/`. Public 5 minutes are for pipeline checks only, not training or tuning. | — | — |
| 2026-11-15 | Osama's registration accepted. Team is now five members. | Team lead | Roles assigned in this README. |
| 2026-11-15 | Strategy: possession from player geometry plus a sticky 3-state decoder (v1). Learned PathCRF-style model is a stretch goal only. | Team | Revisit at the Nov 16 evening check-in. |
| 2026-11-15 | Reused code cloned to `external/`: PathCRF, Metrica sample data. | D | Check Metrica sample licence before use. |

## Important files and links

### In this repository

| File / folder | Purpose | Notes |
|---|---|---|
| [MORE_THAN_A_HACK_2027.md](MORE_THAN_A_HACK_2027.md) | Event facts and participation guide | Public event summary; not official rules |
| [`challenge/`](challenge/README.md) | Official challenge repo: specs, evaluation, starter baselines, public 5-minute ground truth | Match data is separately licensed and is not included |
| `external/` | Cloned reused repos and external data | Git-ignored. Do not commit |
| `src/` | Our pipeline code (features, decoder, rules, submission) | Created as modules land |
| `notebooks/` | Exploratory analysis | Keep outputs small |
| `docs/` | Technical document, schema notes, decisions | No restricted data |
| `submissions/` | Generated submission files | Follow the official submission rules |
| `data/` | Local match data | Git-ignored. Do not commit or redistribute |

### Official resources

- [More Than a Hack 2027 event page and registration link](https://landing.mobileworldcapital.com/more-than-a-hack-2027)
- [Talent Arena](https://talentarena.tech)
- [Hackathon organizer contact](mailto:hackathon.talentarena@mobileworldcapital.com)
- [Challenge repository and rules](https://github.com/metrica-sports/more-than-a-hack-2027)
- [Challenge data folder](https://drive.google.com/drive/folders/1oiEl5sE3yIElIPluOwj5J1uYgFM604uh?usp=drive_link)
- [Challenge README (local clone)](challenge/README.md)

## Team roster

Keep this roster current. The event requires 3–5 team members. Do not store passport numbers, personal phone numbers, or other sensitive registration details here. This roster includes team email addresses, so keep the repository private or remove those addresses before making it public.

| Name | Academic level | Program | University | Email | Role | Registration complete? |
|---|---|---|---|---|---|---|
| Abd Alrahman Ismaik | Undergraduate, Senior | Computer Science | Khalifa University | [100064692@ku.ac.ae](mailto:100064692@ku.ac.ae) | A: Data, baseline, validation | ☑ |
| Faisal Alzarooni | Undergraduate, Senior | Computer Science | Khalifa University | [100064794@ku.ac.ae](mailto:100064794@ku.ac.ae) | B: Core decoder | ☑ |
| Mohamed Elashmouney | Undergraduate, Senior | Applied Math, Statistics & Data Science | Khalifa University | [100064807@ku.ac.ae](mailto:100064807@ku.ac.ae) | C: Features and statistics | ☑ |
| Hardik Sharma | Graduate | Master's, Computer Science | Khalifa University | [100069933@ku.ac.ae](mailto:100069933@ku.ac.ae) | D: Smart Tagging, restarts, reused code | ☑ |
| Osama Qa'adan | Undergraduate, Senior | Computer Science | Khalifa University | [100064881@ku.ac.ae](mailto:100064881@ku.ac.ae) | E: Repo, reproducibility, document | ☑ (accepted Nov 15) |

## Data and repository handling

- Follow the official challenge rules, participant terms, and data-use instructions.
- Keep raw match video, tracking data, and other restricted assets out of Git. Store and share them only through organizer-approved methods.
- Never commit credentials, tokens, passport details, personal contact information, or private registration records.
- Add generated files, `data/`, and `external/` to `.gitignore`.
- Do not upload any video, dataset, or extract to a cloud LLM.
- Record data assumptions and analysis decisions without copying restricted source data into documentation.

## Working agreements

- Keep the task tracker and progress dashboard current.
- Give every active task one owner and a clear completion condition.
- Share code through reviewed commits or pull requests.
- Document the command needed to reproduce important outputs.
- Raise blockers early and record the next action.
- Check official participant materials whenever a rule or technical detail is uncertain.

---

*Event information in this README is based on the public official page and organizer emails. Organizer instructions and updates take precedence.*
