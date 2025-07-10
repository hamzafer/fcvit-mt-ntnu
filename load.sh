#!/bin/bash

echo ">>> Starting load.sh script..."

# Purge and load modules
echo ">>> Purging modules..."
module purge

echo ">>> Loading Anaconda3/2024.02-1..."
module load Anaconda3/2024.02-1
echo ">>> Loading CUDA/12.4.0..."
module load CUDA/12.4.0

# Try activating environment
echo ">>> Trying to activate conda environment..."
conda activate /cluster/home/akmarala/fcvit_env || echo "Conda activate failed"

# Final confirmation
echo ">>> Python version:"
python --version || echo "Python not found"

echo ">>> Done."
