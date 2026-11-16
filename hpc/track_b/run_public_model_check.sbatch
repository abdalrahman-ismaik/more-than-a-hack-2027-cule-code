#!/bin/bash
# Inference plus a public-window pipeline sanity score; no tuning is performed.
# Submit with: sbatch pipeline/run_public_model_check.sbatch
#SBATCH --job-name=public-model-check
#SBATCH --partition=gpu-student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/student_storage/sdp1_group3/public-model-check-%j.out

set -euo pipefail
D=/data/student_storage/sdp1_group3
cd "$D/mtah"
export OMP_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$D/py312/bin/python" -u -m pipeline.predict_public_checks
"$D/py312/bin/python" evaluation/run_evaluation.py \
  --gt public \
  --results evaluation/public_model_checks \
  --out evaluation/public_model_checks/report \
  --bootstrap 1000
