#!/bin/bash

# Run the Insanely Fast Whisper API with optimized memory settings
export USE_FLASH_ATTENTION=true
export WHISPER_DEVICE=cuda:0

# Limit GPU memory usage to 60%
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run the application
python -m uvicorn app.app:app --host 0.0.0.0 --port 8887 --log-level info --workers 1