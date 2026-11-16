# HPC environment and working notes

Updated 2026-11-16 after the LightGBM/GRU retrain, XGBoost CV, and public-window sanity run. Do not
store passwords or private key material here.

## Access

- Login host: `admin`, `10.11.45.11`, account `sdp1_group3`; use KU VPN off campus.
- PuTTY is the known working interactive route. PowerShell OpenSSH key authentication was refused
  during the recent checks. Do not put passwords in scripts or shell history.
- The user's project workspace is `/data/student_storage/sdp1_group3` (`$D` below). Do not access
  other groups' workspace or root-owned paths.

## Hardware and scheduler

- Login node: 256 CPU cores, about 503 GB RAM, and 4 NVIDIA H100 NVL GPUs.
- Slurm is installed (`/usr/bin/sbatch`). `gpu-debug` allows up to 1 hour; `gpu-student` allows up
  to 6 hours. Student GPU jobs are limited to 2 GPUs. Confirm present limits with `sinfo` before a
  new job.
- Submit GPU and long-running work with `sbatch`; avoid heavy runs on the shared login shell. Use a
  small CPU thread count for GPU training and keep job logs under the user's `/data` workspace.

## Storage and data

| Path | Notes |
|---|---|
| `/` | 98 GB filesystem; 5.8 GB free at the 2026-11-15 check. Avoid large writes here. |
| `/data` | 14 TB shared filesystem; about 13 TB free at the 2026-11-15 check. |
| `/data/student_storage/sdp1_group3` | User-owned project workspace. Store code, environments, caches, models, and logs here. |
| `/data/shared_envs/gpu_pytorch` | Shared and read-only for this account; do not install into it. |

The tactical match source is at `$HOME/mtah_data/tactical/BAR-ATH_tactical_atd.xml`. Prediction
jobs need `MTAH_DATA_DIR="$HOME/mtah_data"` exported into the Slurm job environment. For example:

```bash
D=/data/student_storage/sdp1_group3
cd "$D/mtah"
export MTAH_DATA_DIR="$HOME/mtah_data"
sbatch --export=ALL,MTAH_DATA_DIR="$MTAH_DATA_DIR" pipeline/run_public_model_check.sbatch
```

The feature cache is at `mtah/data/cache/features/`: 81 files for 27 games, three realisations per
game, with 108 features. Match data and generated model files must not be committed to Git.

## Python environment

- Private Python environment: `/data/student_storage/sdp1_group3/py312`; invoke it as
  `"$D/py312/bin/python"`. The earlier `conda312` path does not exist.
- Set pip temporary and cache paths to `$D/tmp` and `$D/pipcache`. Do not install into system
  Python, conda base, or the shared PyTorch environment.
- XGBoost 3.4.1 was installed as `xgboost-cu12` in `py312`. Its CUDA smoke test passed on an H100
  with driver 560.35.03 (reported CUDA 12.6).

## Training run record

- The CPU LightGBM + GRU retrain completed. Its OOF result was spellF1 0.5675 and macroF1 0.8623
  over 27 leave-one-game-out folds (7 DFL + 20 SkillCorner).
- Slurm job 672 completed XGBoost's 27 OOF folds and final fit. Keep its outputs separate under
  `mtah/models_xgb/`; it did not overwrite `mtah/models/`.
- Slurm job 675 generated and scored four model variants on the public five-minute window after
  `MTAH_DATA_DIR` was exported. This was a pipeline sanity check only; do not use those labels for
  model tuning or selection. Scores and caveats are recorded in the project README.
