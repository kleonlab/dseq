#!/bin/bash

#SBATCH --job-name=train_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=gpu  # Adjust if your cluster uses a different GPU partition
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

# Change to working directory
cd ~/dseq

# Activate virtual environment
source .venv/bin/activate

# Run the script
python model/train_test.py
