#!/bin/bash
# Batch job for the KU H100 teaching server. Submit from the project folder with:
#   sbatch hpc/run_pipeline.sbatch
#
# The v1 pipeline is CPU-only (numpy), so it does not need a GPU. Ask the server admins which
# partition is the CPU partition and put its name below. Do not run the pipeline in the login shell.
#SBATCH --job-name=culecode-v1
#SBATCH --partition=CPU_PARTITION_NAME
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

mkdir -p logs

# Shared Conda/PyTorch environment provided on the server (numpy is included).
. /etc/profile.d/conda.sh
conda activate /data/shared_envs/gpu_pytorch

# Point at the match data on the server. Data must not be committed to the repo.
export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/mtah_data}"

python -m src.pipeline --name CuleCode_v1
