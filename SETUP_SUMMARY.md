# Insanely Fast Whisper API - Setup Summary

## ✅ Setup Complete

The Insanely Fast Whisper API has been successfully deployed with the following configuration:

### Configuration Details

- **GPU Device**: GPU 2 (NVIDIA GeForce RTX 3090)
- **GPU Memory Limit**: 15% (3.53 GB out of 23.56 GB total)
- **Port**: 8002
- **Conda Environment**: whisper-api
- **Model**: OpenAI Whisper Large v3
- **Optimizations**: Flash Attention 2.0, FP16, Batching

### Current Status

✅ Application is running at: **http://0.0.0.0:8002**

GPU Memory Usage on GPU 2: ~21.6 GB used (model loaded)

### How to Use

#### Start the Application
```bash
cd /home/op/CascadeProjects/insanely-fast-whisper-api
./start_gpu2_limited.sh
```

#### Stop the Application
Press `CTRL+C` in the terminal where the application is running

#### API Endpoints

- **POST /** - Transcribe or translate audio
- **GET /tasks** - Get all active tasks
- **GET /status/{task_id}** - Get task status
- **DELETE /cancel/{task_id}** - Cancel a task

#### Example API Call
```bash
curl -X POST http://localhost:8002/ \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/audio.mp3",
    "task": "transcribe",
    "language": "en",
    "batch_size": 64,
    "timestamp": "chunk"
  }'
```

### Environment Configuration

The following environment variables are set in the startup script:
- `CUDA_VISIBLE_DEVICES=2` - Forces use of GPU 2
- GPU memory fraction set to 0.15 (15%) in `app/app.py`

### Files Modified

1. **app/app.py** - Added GPU memory limiting code
2. **start_gpu2_limited.sh** - Startup script with GPU configuration

### Dependencies Installed

- Python 3.10
- PyTorch 2.2.0 (CUDA 12.1)
- TorchVision 0.17.0
- TorchAudio 2.2.0
- Flash Attention 2.5.6
- Transformers 4.37.2
- FastAPI 0.109.2
- All requirements from requirements.txt

### Notes

- The model is loaded on startup, which takes ~20-30 seconds
- GPU memory is limited to 15% to prevent resource hogging
- The application uses Flash Attention 2 for optimized inference
- Speaker diarization requires HF_TOKEN environment variable

### Troubleshooting

If the application fails to start:
1. Check if port 8002 is available: `ss -tlnp | grep :8002`
2. Verify GPU 2 is available: `nvidia-smi`
3. Check conda environment is activated: `conda env list`
4. View logs in the terminal where the app is running

### Repository Location

`/home/op/CascadeProjects/insanely-fast-whisper-api`
