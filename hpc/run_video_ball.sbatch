#!/bin/bash
# Full-match ball detection on the tactical video, on one GPU of the KU H100 teaching server.
# Submit from the project folder with:  sbatch hpc/run_video_ball.sbatch
#
# Put the detector weights at models/yolo-football-ball-detection.pt on the server first (they are
# not in git). The match videos and tracking are under MTAH_DATA_DIR, not in the repo.
# gpu-student allows up to 2 GPUs and 6 hours for students. This job uses one GPU.
#SBATCH --job-name=culecode-video-ball
#SBATCH --partition=gpu-student
#SBATCH --qos=gpu-student
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# logs/ must exist BEFORE sbatch is run (Slurm opens the output file first). Run: mkdir -p logs results

mkdir -p results

. /etc/profile.d/conda.sh
conda activate /data/shared_envs/gpu_pytorch

# Install ultralytics and pandas only if the shared environment lacks them. Installing torch again
# would replace the server's CUDA build, so do not force a reinstall.
python -c "import ultralytics, pandas" 2>/dev/null || python -m pip install --user -q ultralytics pandas
python -c "import ultralytics, pandas" || { echo "ultralytics/pandas missing: install failed"; exit 1; }

# Fail fast if the job did not get a GPU, instead of running for hours on the CPU.
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "CUDA is not available in this job"; exit 1; }

export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/mtah_data}"

nvidia-smi
# Graded halves only (DATA.md). Ungraded frames are never scored, so they are skipped.
python -m src.video_ball --start 11185 --stop 90652 --device 0 --out results/video_ball_h1.csv
python -m src.video_ball --start 110808 --stop 190198 --device 0 --out results/video_ball_h2.csv
