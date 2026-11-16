#!/bin/bash
# Full leave-one-game-out XGBoost CV and final fit. Writes only to models_xgb/.
# Submit with: sbatch pipeline/run_xgb_all.sbatch
#SBATCH --job-name=xgb-cv-all
#SBATCH --partition=gpu-student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/student_storage/sdp1_group3/xgb-all-%j.out

set -euo pipefail
D=/data/student_storage/sdp1_group3
cd "$D/mtah"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$D/py312/bin/python" -u -m pipeline.train_xgb --all
