# Track B: XGBoost experiment scripts

These files were used for the GPU XGBoost comparison on the KU H100 cluster. The Python modules
are designed to live in the existing MTAH project's `pipeline/` package alongside `data.py`,
`train.py`, `features.py`, `decode.py`, `seq.py`, and `predict.py`. The Slurm files target the
existing cluster layout (`/data/student_storage/sdp1_group3/mtah` and the private `py312` runtime).

## What is included

- `train_xgb.py`: 27-fold leave-one-game-out XGBoost training and final fit; reads the existing
  feature cache, uses the same training data preparation and stride as `pipeline.train`, and writes
  only to `models_xgb/`.
- `run_xgb_fold.sbatch`: one-fold sanity job.
- `run_xgb_all.sbatch`: full OOF CV plus final fit.
- `predict_public_checks.py` and `run_public_model_check.sbatch`: inference for the four recorded
  variants and the separate public-window sanity evaluation. Public labels must remain separate
  from model fitting, decoder tuning, and model selection.

The training source requires the existing project `pipeline/` package and a CUDA-enabled XGBoost
installation. These scripts do not include feature caches, match data, trained models, or evaluation
outputs. Those artifacts remain outside Git.

The cluster copies used for the November 2026 run are already in place. For a fresh deployment,
copy the two Python modules to the cluster's `mtah/pipeline/` directory and the three Slurm/helper
files to the corresponding project location before submitting. Submit the fold check before the full
run, and inspect the output/logs before proceeding.
