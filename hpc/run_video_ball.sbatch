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

mkdir -p logs results

. /etc/profile.d/conda.sh
conda activate /data/shared_envs/gpu_pytorch

# ultralytics is not in the shared environment by default; install it into your own user site.
python -m pip install --user -q ultralytics pandas

export MTAH_DATA_DIR="${MTAH_DATA_DIR:-$HOME/mtah_data}"

nvidia-smi
python -m src.video_ball --device 0 --out results/video_ball_full.csv
