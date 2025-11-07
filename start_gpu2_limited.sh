#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper-api

# Set GPU 2 as the visible device
export CUDA_VISIBLE_DEVICES=2

# Set GPU memory configuration for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the application with uvicorn
echo "Starting Insanely Fast Whisper API on GPU 2 with limited utilization..."
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "Max GPU Utilization: 15%"

uvicorn app.app:app --host 0.0.0.0 --port 8002 --reload
