#!/bin/bash

#SBATCH --job-name=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

cd ~/dseq
source .venv/bin/activate
uv run scripts/gpu_test.py