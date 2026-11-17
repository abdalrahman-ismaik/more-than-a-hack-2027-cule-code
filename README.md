# CuleCode - More Than a Hack 2027

Final Initial Challenge submission by **CuléCode**, Khalifa University, Abu Dhabi.

**Team:** Abd Alrahman Ismaik, Faisal Alzarooni, Mohamed Elashmouney, Hardik Sharma,
and Osama Qa'adan. Contact: abdalrahman.ismaik@gmail.com.

The final model is a DFL-trained LightGBM + bidirectional GRU ensemble followed by a
three-state Viterbi decoder. It predicts home possession (H), away possession (A),
and dead ball (D); frames outside the supplied halves are X. There are **108 features**.

## Submission files

- [Possession CSV](submissions/CuleCode.csv)
- [Technical document (PDF)](submissions/CuleCode.pdf)
- [Editable technical document](docs/CuleCode.md)
- [Checksums and verification](submissions/manifest.json)

The CSV and PDF use the team's established ASCII filename spelling, `CuleCode`.

## Reproduce the submitted CSV

Use Python **3.12** and the pinned CPU dependencies. The weights are included; training
and external training downloads are unnecessary for prediction.

```bash
python -m venv .venv
# Activate .venv using your platform's command, then:
python -m pip install -r requirements.txt
```

Put the organizer data under `data/tactical/`, or set `MTAH_DATA_DIR` to the directory
containing `tactical/`. Only `BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt` are read. Never commit the supplied match data.

PowerShell:

```powershell
$env:MTAH_DATA_DIR = "D:\match-data"
python run.py predict
```

Linux/macOS:

```bash
export MTAH_DATA_DIR=/path/to/match-data
python run.py predict
```

This writes `submissions/CuleCode.csv` and runs the official validator. Prediction uses
CPU inference. The first run also caches parsed tracking under ignored `data/cache/`.
Allow several minutes and several GB of available RAM for the original text loader.
The organizer-published pre-kickoff interval 110808-113741 is fixed to dead-ball evidence before decoding, as in the team's latest main. The trained weights remain unchanged.
The manifest records the tested environment, exact artifact hashes and verification.

```bash
python evaluation/validate_submission.py submissions/CuleCode.csv
python evaluation/run_evaluation.py --help
```

The scorer's public five-minute ground truth is for pipeline checks only. No target-match
labels enter model fitting or decoder selection. Public-window scores are not full-match
performance estimates.

## Training provenance

Seven other DFL/IDSSE Bundesliga and 2. Bundesliga matches supply provider possession and
ball-status labels, under CC BY 4.0. Training simulates automatic-tracking noise and uses
team-swap augmentation. Decoder costs were selected on external out-of-fold predictions.
Recorded DFL validation: spellF1 **0.6900**, macroF1 **0.9178**. These figures include decoder
selection on those same folds and are not an untouched test estimate. Original OOF arrays
are not distributed; model artifacts are the original committed teammate checkpoint.

To rebuild training in a separate checkout (this overwrites `models/`):

```bash
python run.py download
python run.py train
python run.py predict
```

The downloader uses the public `pysport/idsse-data` mirror. A complete retrain was not rerun
for this release; exact prediction reproduction from the frozen weights is the release check.

## Repository layout

| Location | Contents |
| --- | --- |
| `pipeline/` | Final model: data, noise, features, training, sequence model, decoding, prediction |
| `models/` | Only the selected weights, decoder and supporting metadata |
| `evaluation/` | Official scorer, validator and public five-minute labels |
| `docs/` | Final technical document and source acknowledgements |
| `tools/` | Technical PDF builder |
| `submissions/` | Final CSV and verification manifest |

Earlier heuristic, detector, mixed-dataset and XGBoost work is archived separately and is
not part of the final model.

## Sources and license

- [Official challenge](https://github.com/metrica-sports/more-than-a-hack-2027)
- [Osama's source implementation](https://github.com/OsamaQ05/barca-hack-2027), commit
  `0fe3337c1b0d861fa3f5d61630687232220ff5ad`
- [DFL/IDSSE public training-data mirror](https://huggingface.co/datasets/pysport/idsse-data)

See [acknowledgements](docs/ACKNOWLEDGEMENTS.md) and [LICENSE](LICENSE). The MIT license
covers code and documentation as specified there; restricted match assets and external
training data retain their separate terms.
