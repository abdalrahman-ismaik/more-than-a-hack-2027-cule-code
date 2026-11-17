# CuléCode
## Possession from automatic tracking
More Than a Hack 2027 - Initial Challenge | Technical document | 17 November 2026

**Team:** Abd Alrahman Ismaik, Faisal Alzarooni, Mohamed Elashmouney, Hardik Sharma and Osama Qa'adan.

**Contact:** abdalrahman.ismaik@gmail.com

### 1. Final approach

We infer home-team possession (H), away-team possession (A) and dead ball (D) from the supplied tactical automatic tracking. Frames outside the supplied halves are X. The final submission has 199,319 rows at 25 frames per second.

The selected system is a LightGBM plus bidirectional GRU ensemble trained on seven external DFL matches. Both models produce probabilities for H, A and D. We average their probabilities and decode each half using a three-state Viterbi algorithm. The model uses 108 geometric and temporal features and can use future observations: this is an offline labeling system, not a live predictor.

Only the original DFL-trained checkpoint is used in the submitted file. It comes from Osama Qa'adan's repository, commit 0fe3337c1b0d861fa3f5d61630687232220ff5ad. The selected weights, decoder and feature metadata are unchanged. Packaging adds a team-named command, validation and a feature-order check. We retain the organizer-published second-half pre-kickoff rule already added in the team repository: frames 110808 through 113741 receive dead-ball evidence before decoding.

### 2. Assets and training labels

**Target-match inputs:** tactical ATD metadata and player/ball coordinates. Team membership comes from metadata. Aggregate tracking statistics, computed without labels, parameterize the training noise simulation. Tactical video, broadcast footage, homography and Smart Tagging are not prediction inputs to the final model. Smart Tagging was considered as a consistency check, not a fitted prior.

**External training data:** seven DFL/IDSSE Bundesliga and 2. Bundesliga matches, with provider BallStatus and BallPossession flags at 25 Hz, distributed under CC BY 4.0. Match IDs are J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ and J03WR9. Labels come from the provider, not team annotations: D for dead ball, otherwise H or A from possession. No target-match labels enter model fitting.

A training-target rule folds opposing spells shorter than one second into the previous team's possession when followed by that team again or by dead ball. Short spells immediately after restarts are retained. This is an operational approximation to controlled possession, not independently verified touch annotation.

<!-- pagebreak -->

### 3. Noise adaptation and features

Clean DFL tracking differs from the supplied automatic tracking. The training pipeline simulates ball gaps using speed-conditioned seen/missing transitions, player dropout, position jitter where needed, and ball motion during otherwise frozen dead-ball spans. Noise parameters are aggregate rates and quantiles; the repository does not include target per-frame tracking.

Features are independent of persistent player identity. They summarize ball location, speed, visibility and gap duration; nearest and second-nearest player distances; player counts near the ball; team shape and motion; and previous/next control evidence. A control cue requires a player within 1.5 m, at least 1 m closer than the nearest opponent, with relative speed below 3 m/s.

Single-frame ball spikes are removed and gaps are interpolated. Time since/until observation and gap length let the model discount uncertain interpolated coordinates. Temporal means include past and future windows up to ten seconds. Temporal processing stays within each half. Both team orientations are used during fitting and prediction, with home/away probabilities swapped back before averaging.

### 4. Models, decoding and possession rules

**LightGBM:** three-class gradient boosting, 400 rounds and 63 leaves, trained with a fixed seed and deterministic settings. Training uses every third labeled frame, two noise realizations per game and both team orientations.

**GRU:** a two-layer bidirectional recurrent network with hidden size 64, evaluated at 5 Hz from the same features and interpolated back to 25 Hz. Training uses six epochs and roughly 51-second chunks. The committed artifact includes its feature scaler.

**Ensemble:** average the team-symmetric LightGBM and GRU H/A/D probabilities. Decode the resulting log probabilities separately within the organizer-defined half spans. Staying in a state costs zero; a direct team switch costs 4, entering dead ball costs 2 and leaving dead ball costs 48. These are log-score penalties, not probabilities. The organizer FAQ explicitly identifies frames 110808 through 113741 as dead ball. As in the team repository, their probabilities are set to (0.000001, 0.000001, 0.999998) before decoding, so the surrounding sequence is decoded normally. This published boundary rule is not learned from public-window labels.

**Loose balls and challenges:** the features and sequence decoder can retain the prior possessor until control changes; closest-player identity is not a direct final label. **Stoppages:** D separates possessions, including two spells of the same team before and after a stoppage. **Restarts:** a short first possession after D is retained in training. **Missing ball:** missingness alone does not force D or X. **Outside halves:** write X; the pipeline predicts H, A or D throughout each supplied half, including graded pre-kickoff frames.

<!-- pagebreak -->

### 5. Evaluation and model selection

The primary metric, spellF1, matches same-team possession intervals one-to-one using temporal intersection-over-union and averages F1 over thresholds 0.3, 0.4, 0.5, 0.6 and 0.7. MacroF1 averages frame classification F1 over H, A and D. Correct frame totals do not guarantee correctly recovered possessions.

**Recorded external validation:** seven-game leave-one-game-out training produced mean spellF1 0.6900 and macroF1 0.9178. Recorded per-game spellF1 values are 0.631, 0.675, 0.730, 0.682, 0.706, 0.695 and 0.712. Decoder costs were selected on these same out-of-fold predictions, so this is a development estimate with selection optimism, not an untouched test score. The original OOF arrays were unavailable for independent rescoring; checkpoint identity and recorded logs were checked.

**Public five-minute check:** the recovered original prediction scored spellF1 0.6452 and macroF1 0.8773 on 7,500 public frames, with 14 predicted versus 17 reference possessions. The kickoff rule concerns only the second half and leaves this first-half check unchanged. This small window checks pipeline operation and is not a full-match performance estimate. It was not used to train the models or choose decoder settings.

**Other work:** earlier geometry-and-tag heuristics, optional video-ball filling, a 27-game DFL/SkillCorner retrain and XGBoost variants were explored. On a common 27-game evaluation, LightGBM + GRU scored 0.5675 spellF1 versus 0.5388 for XGBoost, 0.5472 for XGBoost + GRU and 0.5620 for an equal three-model blend. The retrain's DFL subset scored 0.6689. These mixed-dataset results are not directly comparable to the original seven-game mean because preprocessing, noise realizations, training sets and decoder selection differ. The final release retains the packaged DFL checkpoint; the other models and detector are excluded.

### 6. Limitations

The ball is missing on about 41% of graded target frames. Noise simulation cannot fully reproduce the errors of another tracking provider, and interpolated positions can be misleading during long gaps. DFL possession conventions differ from the challenge's controlled-possession definition; the one-second training rule is only an approximation.

Stationary live play, goalkeeper control and whistle stoppages can be confused. Tracking alone does not reveal the referee's whistle. Global transition penalties can merge short possessions or delay boundaries. Seven external matches provide limited coverage, and the external validation shares its data with decoder selection. The private full-match score is unknown.

<!-- pagebreak -->

### 7. Reproduction and release verification

Use Python 3.12 with the pinned CPU packages in requirements.txt. The release was tested with Python 3.12.12. Install dependencies, set MTAH_DATA_DIR to the folder containing tactical/, and run:

`python run.py predict`

This loads the committed models, writes submissions/CuleCode.csv, and runs the official validator. Only the supplied tactical ATD XML and TXT are needed. The first run builds a local parsed-tracking cache. The generated labels must reproduce from a clean checkout; hashes, exact tested package versions and check outcomes are recorded in submissions/manifest.json.

The CSV has a frame,label header, frames 0 through 199318 in order and labels H/A/D/X. The validator checks structure rather than prediction accuracy. No hand-edited labels are applied. The exact submission files are CuleCode.csv and CuleCode.pdf.

Training code is included. In a separate checkout, python run.py download retrieves external DFL XML files and python run.py train rebuilds models and selects the decoder. A complete retrain was not rerun for this release; reproduction of the submitted CSV uses the frozen artifacts. CPU/runtime differences may affect a new training run.

### 8. Sources, licensing and assistance

**Official challenge, scorer and configuration:** https://github.com/metrica-sports/more-than-a-hack-2027. The pre-kickoff rule is published in FAQ.md, under "Which frames are actually graded?"

**Original model implementation and frozen artifacts:** https://github.com/OsamaQ05/barca-hack-2027 at commit 0fe3337c1b0d861fa3f5d61630687232220ff5ad.

**External training data:** DFL / Sportec Solutions IDSSE open data, CC BY 4.0, via https://huggingface.co/datasets/pysport/idsse-data. Provider flags supply the labels. No labels from the target match are used for training.

The repository retains the upstream MIT notice. Restricted challenge assets are not redistributed. External training data and software libraries retain their own licenses. See docs/ACKNOWLEDGEMENTS.md for provenance.


