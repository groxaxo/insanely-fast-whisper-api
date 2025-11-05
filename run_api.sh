#!/bin/bash

# Script to run the insanely-fast-whisper-api Docker container with optimized memory settings

echo "Checking if insanely-fast-whisper-api container is already running..."
running_container=$(docker ps --format "{{.Names}}" | grep "insanely-fast-whisper-api")

if [ -n "$running_container" ]; then
    echo "Container $running_container is already running."
    echo "To access the running container, use: docker exec -it $running_container bash"
else
    echo "Container is not running. Starting the insanely-fast-whisper-api container with optimized settings..."
    # Run the container in detached mode with GPU access and optimized memory settings
    docker run -d --gpus all --name insanely-fast-whisper-api-container \
        -p 8887:8887 \
        -e USE_FLASH_ATTENTION=true \
        -e WHISPER_DEVICE=cuda:0 \
        -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
        -v $(pwd)/data:/data \
        yoeven/insanely-fast-whisper-api:latest
        
    if [ $? -eq 0 ]; then
        echo "Container started successfully!"
        echo "Access the API at: http://localhost:8887"
        echo "To view logs: docker logs -f insanely-fast-whisper-api-container"
        echo "To stop the container: docker stop insanely-fast-whisper-api-container"
    else
        echo "Failed to start the container."
        exit 1
    fi
fi