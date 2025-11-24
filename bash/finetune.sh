#!/bin/bash

#SBATCH --job-name=my_training
#SBATCH --time=08:00:00           # 4 hours (adjust as needed)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --gres=gpu:4               # Request 1 GPU
#SBATCH --mem=32G                  # Memory
#SBATCH --output=train_%j.out      # Standard output log
#SBATCH --error=train_%j.err       # Standard error log

# Change to working directory
cd ~/dseq

# Activate virtual environment
source .venv/bin/activate

# Run the script
python scripts/finetune.py
