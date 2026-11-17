# CuléCode: possession from automatic tracking

*More than a Hack 2027, Phase 1: technical document. Team CuléCode, Khalifa University.*

## 1. Approach

We label every frame `H`, `A` or `D` inside the two graded halves, and `X` outside them.

The ground truth is mechanical. A frame is `D` where the *corrected* tracking marks the ball dead;
otherwise it is the team that tracking names as owning the ball (CHALLENGE.md). Open data with
exactly these two per-frame flags exists for other matches: the DFL / IDSSE release, with
`BallStatus` and `BallPossession` on every frame of seven Bundesliga and 2. Bundesliga games. So we
treat the task as supervised sequence labelling:

1. **Train on other matches.** We use the seven DFL games, after degrading their clean tracking to
   match the noise of the provided automatic tracking.
2. **Two models, averaged.** A gradient-boosted classifier (LightGBM) and a bidirectional GRU
   predict H/A/D per frame from identity-free, team-symmetric features. The features look both
   backwards and forwards in time.
3. **Decode possessions.** A three-state Viterbi decoder turns the per-frame probabilities into
   possessions. Its switching costs maximise `spellF1` on held-out DFL games.

Held-out score on DFL games (leave-one-game-out): **spellF1 0.690, macroF1 0.918** (§6). On the
public five minutes (17 possessions, a pipeline check only) the file scores spellF1 0.645, macroF1
0.877, against 0.33 for the published baseline.

No labels of BAR-ATH are used anywhere. The only BAR-ATH-derived quantities are label-free
aggregate statistics of its tracking and Smart Tagging, used to make the training data look like
it (§4, §8). The public five minutes were used only as a pipeline check: frame alignment, H/A
orientation and file format.

**Why this rather than something else.** The team first built a hand-weighted pipeline. A control
score from the nearest home and away player, Smart Tagging priors and a sticky decoder had their
few weights chosen on other matches. It reached spellF1 0.41–0.47 on held-out SkillCorner and
Metrica games, and 0.31 on the public window. Learning from data that carries the exact target
labels was far stronger, so the learned pipeline is the submission. The hand-built pipeline stays
in `src/` as a documented alternative.

## 2. Assets used

| asset | used? | how |
|---|---|---|
| Tactical ATD (players + ball) and metadata | yes | the main input at prediction time; team of each track slot |
| Smart Tagging | tested, see §8 | SET PIECES as restart evidence, quick and learned versions |
| Tactical video | no (for the submission) | a YOLO ball detector was run on it as an experiment (§8) |
| Homography | no (for the submission) | maps video detections to the pitch in that experiment |
| Broadcast video / ATD | no | not synchronised to the tactical video; the tactical ATD covers the whole match |
| Public 5-minute ground truth | pipeline check only | never used to choose a model, setting or version |
| FAQ.md frame table | yes | frames 110,808–113,741 (second-half kickoff wait) are graded and all `D` (§7) |

## 3. Training data

All labelled data comes from other matches. None of it is about BAR-ATH, and none of it was
labelled by us.

| dataset | source, licence | labels, who made them | how much | used for |
|---|---|---|---|---|
| **DFL / IDSSE open data** | Bassek et al., *Scientific Data* 2025; DFL and Sportec Solutions; CC-BY 4.0; Hugging Face mirror `pysport/idsse-data` | per-frame `BallStatus` and `BallPossession` from the data provider's corrected tracking; set-piece events from the official event log | 7 matches, ~1.0 M frames at 25 fps | **training and validation of the submitted model** |
| SkillCorner open data | `SkillCorner/opendata`, MIT | possession spans from the published phases of play (SkillCorner) | 20 A-League matches, 10 fps | validation of the hand-built pipeline; a dropped training experiment (§8) |
| Metrica sample data | `metrica-sports/sample-data` | possession proxy we derived from the event log | 2 matches | weight selection for the hand-built pipeline only |
| YOLO11n ball detector | `martinjolif/yolo-football-ball-detection`, AGPL-3.0 | human-labelled ball boxes from its authors | pretrained, not fine-tuned | the dropped video-ball experiment only (§8) |

**Labels.** `D` if the ball is dead, otherwise `H` or `A` from the possession flag. This is the same
construction CHALLENGE.md describes for the ground truth.

**One change to the target, taken from the written definition.** The DFL possession flag flips on
any touch. CHALLENGE.md says a touch without control, a deflection, or an aerial duel won without
subsequent control is *not* a change of possession. So we fold back a spell by the other team
that is shorter than 1 s and is followed by the original team again, or by a dead ball. A short
spell straight after a restart is kept.

**Two problems in the raw DFL feed that we fixed:**
- The referees are tracked with their own team id. We keep only the two clubs.
- The ball position is frozen on every dead-ball frame. A model trained on that learns "ball
  perfectly still ⇒ dead", a cue that never occurs in automatic tracking. We replace each frozen
  span with a smooth drift plus a random wander.

## 4. Noise adaptation

The provided tracking is automatic: the ball is missing on 41% of graded frames, players drop out
and tracks fragment. The DFL tracking is corrected and complete, so we degrade the training games
until their inputs look like the test game's. Every statistic below is an aggregate measured on
the provided tracking without labels, which RULES.md allows ("learning without human labels").
They are saved in `models/possession/noise_stats.json`.

| effect | how it is simulated | matched statistic |
|---|---|---|
| ball missing | two-state Markov chain (seen/missing); its per-frame rates depend on a collective-speed bin (10 s mean player speed, 6 bins) | maximum-likelihood rates per bin on the provided tracking |
| player dropout | each player disappears in geometric runs of ~2 s | mean players tracked per team (10.2) |
| position jitter | Gaussian, only where the test data is noisier | robust std of second differences |
| dead ball | smooth drift plus an Ornstein-Uhlenbeck velocity wander (σ 1.4 m/s, τ 1 s) | ball speed in slow phases |

**Adversarial validation.** A classifier trained to tell DFL frames from BAR-ATH frames exposed
three bugs, all fixed:
- the referees counted as away players;
- a ball-velocity clip;
- non-robust team-shape statistics.

We also checked whether the provided tracking loses the ball more often near players, as
occlusion would suggest. The per-frame loss rate is only about 1.4× higher within 1 m of a player
than our simulation's, and lower at medium range, so we kept the speed-based model.

## 5. Features

There are 108 per-frame features, computed identically for both datasets (`pipeline/features.py`).

- **Identity-free.** Track slots are tracklets, not people, so every feature is a per-team
  aggregate:
  - nearest and second-nearest distance to the ball;
  - players within 3 m and 8 m;
  - the nearest player's speed and velocity relative to the ball;
  - robust team shape: median position, IQR, and 10–90% length and width.
- **Team-symmetric.** Each team's geometry is expressed in its own attacking direction, which is
  inferred per half. Swapping the teams and the H/A labels therefore gives another valid example.
  We use it to double the training data, and at prediction time we average the normal and the
  swapped prediction.
- **Ball handling.** Spikes are removed and the ball is interpolated through gaps. Every frame also
  carries the time since the ball was last seen, the time until it is next seen, and the gap
  length.
- **Control events.** These follow the CHALLENGE.md proxy for control: a player within 1.5 m of
  the ball, at least 1 m closer than any opponent, and moving with it (under 3 m/s relative
  speed). From these come who controlled the ball last and next, and how long ago and how far
  ahead.
- **Offline context.** Past and future window means over 0.5–10 s of the signals above.

## 6. Model, decoder and validation

**Two per-frame models, averaged.**
- **LightGBM**: multiclass H/A/D, 400 rounds, 63 leaves, deterministic mode. It trains on every
  3rd frame of two independent noise realisations per game, in both team orientations.
- **Bidirectional GRU**: 2 layers, 64 units, at 5 fps over 51 s chunks, CPU, deterministic.

Both use test-time team-swap averaging.

**Decoder.** A three-state HMM (H, A, D), decoded exactly with Viterbi. An H↔A switch, entering
`D` and leaving `D` each have a cost. The three costs were chosen by an exhaustive grid of 729
settings, maximising mean `spellF1` from the official scorer over the seven held-out games.

**Validation protocol.** Leave-one-game-out over the seven DFL games. The held-out game uses a
noise realisation that no model was trained on. All numbers are out-of-fold:

| configuration | spellF1 | macroF1 |
|---|---|---|
| ensemble, per-frame argmax (no decoder) | 0.535 | 0.917 |
| LightGBM + tuned Viterbi | 0.659 | 0.910 |
| GRU + tuned Viterbi | 0.690 | 0.912 |
| **LightGBM + GRU + tuned Viterbi (submitted)** | **0.690** | **0.918** |

Per held-out game: 0.631, 0.675, 0.730, 0.682, 0.706, 0.695, 0.712.

These are scores on *other games with simulated noise*. They are not a prediction of the BAR-ATH
score, because of the domain gap: a different league, the provider's own conventions, and real
rather than simulated tracking errors.

**Label-free consistency check on BAR-ATH.** Our predictions agree with Smart Tagging's team tags
on 98% of live frames in Build Up phases, 90% in Progression, and 75% over all team-tagged phases.

## 7. Possession rules implemented

- **Dead ball.** Learned from the provider's alive/dead flag. In practice the model relies on:
  - collective deceleration;
  - the ball leaving the pitch;
  - the ball being unseen while everyone is slow.

  Every stoppage ends a possession.
- **Second-half kickoff wait.** FAQ.md states that frames 110,808–113,741 are graded and all `D`.
  We set the model's probabilities there to `D` before decoding, so the kickoff that follows is
  decoded normally. This removed 12 spurious possessions.
- **Restarts.** After a stoppage, the first team to control the ball takes possession. Short spells
  after a restart are not merged.
- **Loose balls and challenges.** Possession stays with the last team. A spell under 1 s by the
  other team that ends with the original team or a dead ball is not a change: it counts as a
  touch, a deflection, or an aerial duel without control.
- **Changes of possession** need control by the other team. They cost the decoder a switch, which
  is what keeps one possession from splitting into many.
- **Abstention.** We never write `X` on a graded frame.

## 8. What we tried and dropped

All decisions below were made on held-out games from other matches, never on BAR-ATH.

| idea | result | decision |
|---|---|---|
| Hand-built pipeline (control score, Smart Tagging priors, sticky decoder; weights chosen on other matches) | spellF1 0.41–0.47 on held-out SkillCorner / Metrica games | replaced by the learned pipeline |
| Smart Tagging, **quick version**: rules that raise P(dead) before an expected restart and P(tagged team) after it, tuned on DFL with simulated tags | 0.6900 → 0.6915 | not adopted: below any meaningful margin |
| Smart Tagging, **learned version**: tag inputs added to both models, retrained | 0.6900 → 0.6843; worse on 6 of 7 games (LightGBM alone: 0.6608 → 0.6594) | not adopted |
| GRU seed ensemble (seeds 2027 and 1 averaged) | 0.6900 → 0.6895; the second seed alone also scores 0.6900 | not adopted |
| Adding the 20 SkillCorner games to training (27-fold CV) | DFL subset 0.669 vs 0.690 | dropped: different label convention (phases of play) |
| XGBoost in place of, or alongside, LightGBM | 0.606–0.654 on the DFL subset | dropped |
| Self-training on BAR-ATH with confident pseudo-labels | −0.001 to +0.003 | dropped |
| Video ball detection (YOLO11n, homography) where the tracking ball is missing | fills 17.6% of the gaps; 18% of detections disagree with the tracking by > 6 m | not used (below) |

**How the Smart Tagging test works.** The model can only learn how far to trust a tag from
training games that carry tags, and the DFL games carry none. They do carry the official event
log, with every set piece and the moment it was taken, so we simulate Smart Tagging on them. All
inputs to the simulation are aggregate measurements on BAR-ATH, not labels:
- **Which set pieces get a tag:** the per-type rate at which Smart Tagging tags that type in
  BAR-ATH, relative to how often the type occurs in a DFL game. Throw-ins and goal kicks get 0.59,
  corners and kick-offs 1.0, free kicks 0.2.
- **Where the tag sits:** tags start about 4.7 s before the restart our model predicts
  (10–90%: 1–9 s) and last a fixed ~35 s.
- **Tags with no restart nearby:** 14% of tags match no predicted restart, so 14% random tags are
  added.
- **Teams:** throw-ins and kick-offs get no team, as in Smart Tagging.

Few free kicks are tagged, and free kicks are where the model is slowest to see a stoppage. That
limits what the tags can add.

**Video ball.** Our upper-bound experiment shows the ceiling: with the ball never missing (all
other noise kept), the same LightGBM + decoder reaches spellF1 0.711, against 0.654 with realistic
gaps. A detector that fills 18% of the gaps, with some false detections, would recover a small part
of that. We could not measure its effect without labels on BAR-ATH. It would also make the
submission depend on a GPU run over the video, and on an AGPL-3.0 model.

## 9. Limitations

- **Domain gap.** The noise simulation matches aggregate statistics only. An adversarial classifier
  can still tell the two domains apart, so real errors on BAR-ATH can differ from simulated ones.
- **Stoppages for fouls** are detected 1–3 s late: the tracking shows no whistle, only players
  slowing down.
- **A goalkeeper holding the ball** can read as a stoppage (ball unseen, everyone slow). On held-out
  DFL games our model calls only 10% of such holds dead, but the share may be higher on BAR-ATH.
- **Seven training games.** More open tracking data with possession flags would help most. The
  per-game spread (0.63–0.73) shows how much results vary by match.
- **The decoder's costs are global.** A context-dependent switching cost, or a semi-Markov
  duration model, is the next step.

## 10. Reproduction

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027 challenge   # loaders, scorer, validator
pip install -r requirements-pipeline.txt     # Python 3.12, CPU only
# match data under data/ (data/tactical/BAR-ATH_tactical_*), or set MTAH_DATA_DIR
python run.py predict    # ~2 min: submissions/CuleCode.csv from the committed models, then validates it
python run.py all        # from scratch: download DFL open data, features, CV, tuning, training, predict
```

- **Determinism.** Seeds are fixed, LightGBM runs in deterministic mode, and PyTorch runs on the
  CPU with deterministic algorithms. The committed models regenerate the submitted file byte for
  byte (`submissions/CuleCode.csv`, MD5 `03433039381d97ce285e8fef1b7d95cf`). Refitting both final
  models from scratch reproduces the committed files bit for bit.
- **What is committed.** Code, trained models, decoder costs, and aggregate statistics of the
  provided tracking and Smart Tagging (`models/possession/*.json`: rates, quantiles and offsets,
  nothing per-frame). No match data, video or per-frame tracking.

| file | what |
|---|---|
| `pipeline/data.py` | loaders: provided ATD, DFL open data |
| `pipeline/noise.py` | noise measurement and simulation |
| `pipeline/features.py` | per-frame features |
| `pipeline/tags.py`, `pipeline/tag_rules.py` | Smart Tagging restarts, DFL tag simulation, the quick version |
| `pipeline/train.py` | leave-one-game-out CV, decoder tuning, final models |
| `pipeline/seq.py` | bidirectional GRU |
| `pipeline/decode.py` | Viterbi decoder, controlled-possession target |
| `pipeline/predict.py` | the submission file |
| `pipeline/experiments.py`, `pipeline/gru_seeds.py` | dropped experiments: ball upper bound, self-training, GRU seeds |
| `src/` | the hand-built pipeline and the SkillCorner / Metrica / video experiments |

**Use of AI assistance.** The learned pipeline was written with an LLM coding assistant (Claude).
In line with the competition terms, no match video, tracking values or other restricted material
was given to it for the modelling work: it saw only code, error messages and aggregate statistics.
Early on, before we found that clause, one video frame and a few lines of the raw files were
exposed to it. Nothing derived from them is used.
