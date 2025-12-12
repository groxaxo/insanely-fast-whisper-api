```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗███╗   ██╗███████╗ █████╗ ███╗   ██╗███████╗██╗  ██╗   ██╗          ║
║   ██║████╗  ██║██╔════╝██╔══██╗████╗  ██║██╔════╝██║  ╚██╗ ██╔╝          ║
║   ██║██╔██╗ ██║███████╗███████║██╔██╗ ██║█████╗  ██║   ╚████╔╝           ║
║   ██║██║╚██╗██║╚════██║██╔══██║██║╚██╗██║██╔══╝  ██║    ╚██╔╝            ║
║   ██║██║ ╚████║███████║██║  ██║██║ ╚████║███████╗███████╗██║             ║
║   ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝             ║
║                                                                           ║
║              ███████╗ █████╗ ███████╗████████╗                           ║
║              ██╔════╝██╔══██╗██╔════╝╚══██╔══╝                           ║
║              █████╗  ███████║███████╗   ██║                              ║
║              ██╔══╝  ██╔══██║╚════██║   ██║                              ║
║              ██║     ██║  ██║███████║   ██║                              ║
║              ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝                              ║
║                                                                           ║
║                    WHISPER API - BLAZING FAST                            ║
║                  OpenAI-Compatible Speech-to-Text                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

<div align="center">

[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Compatible-76B900?style=for-the-badge&logo=nvidia)](https://www.nvidia.com/)
[![Model](https://img.shields.io/badge/Model-Whisper%20Large%20V3%20Turbo-412991?style=for-the-badge&logo=openai)](https://huggingface.co/openai/whisper-large-v3-turbo)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Transcribe audio to text at blazing fast speeds with OpenAI's Whisper Large V3 Turbo**

[Features](#-features) • [Quick Start](#-quick-start) • [API Usage](#-api-usage) • [Deploy](#-deployment) • [Documentation](#-documentation)

</div>

---

## 🚀 Features

<table>
<tr>
<td width="50%">

### ⚡ Performance
- **2x Faster** than regular Whisper V3
- **Flash Attention 2.0** optimization
- **GPU-accelerated** inference
- **Batch processing** for efficiency
- **~2 min** for 150 min of audio (A100)

</td>
<td width="50%">

### 🎯 Capabilities
- **99+ Languages** auto-detection
- **OpenAI Compatible** API
- **Speaker Diarization** support
- **Async Processing** with webhooks
- **Docker Ready** for easy deployment

</td>
</tr>
</table>

### 🎤 Key Highlights

```
✅ Transcribe audio to text at blazing fast speeds
✅ Fully open source and deployable on any GPU cloud provider
✅ Built-in speaker diarization
✅ Easy to use FastAPI layer
✅ Async background tasks and webhooks
✅ Optimized for concurrency and parallel processing
✅ Task management, cancel and status endpoints
✅ Admin authentication for secure API access
✅ Fully managed API available on JigsawStack
```

---

## 📋 Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull the latest image
docker pull yoeven/insanely-fast-whisper-api:latest

# Run the container
docker run -d \
  --gpus all \
  -p 8000:8000 \
  yoeven/insanely-fast-whisper-api:latest
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/groxaxo/insanely-fast-whisper-api.git
cd insanely-fast-whisper-api

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Flash Attention
pip install wheel ninja packaging
pip install flash-attn==2.5.6 --no-build-isolation

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

### Option 3: Poetry (Development)

```bash
# Install dependencies
poetry install

# Run the API
poetry run uvicorn app.app:app --reload
```

---

## 🔌 API Usage

### Transcribe Audio

**OpenAI-Compatible Endpoint:**

```bash
curl -X POST http://localhost:8000/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "language=en"
```

**Response:**
```json
{
  "text": "Your transcribed text here..."
}
```

### Standard Endpoint

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/audio.mp3",
    "task": "transcribe",
    "language": "en",
    "batch_size": 64,
    "timestamp": "chunk"
  }'
```

**Response:**
```json
{
  "output": {
    "text": "Transcribed text...",
    "chunks": [...]
  },
  "status": "completed",
  "task_id": "uuid-here"
}
```

### Async Processing with Webhook

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/audio.mp3",
    "task": "transcribe",
    "is_async": true,
    "webhook": {
      "url": "https://your-webhook-url.com",
      "header": {
        "Authorization": "Bearer your-token"
      }
    }
  }'
```

### Speaker Diarization

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -H "x-admin-api-key: your-admin-key" \
  -d '{
    "url": "https://example.com/audio.mp3",
    "task": "transcribe",
    "diarise_audio": true
  }'
```

**Note:** Requires `HF_TOKEN` environment variable. See [setup instructions](#speaker-diarization-setup).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/` | Transcribe or translate audio |
| `POST` | `/audio/transcriptions` | OpenAI-compatible transcription endpoint |
| `GET` | `/tasks` | Get all active transcription tasks |
| `GET` | `/status/{task_id}` | Get the status of a task |
| `DELETE` | `/cancel/{task_id}` | Cancel async background task |

### Request Parameters

#### POST `/` - Standard Endpoint

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | **required** | URL of audio file |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `language` | string | `None` | Language code (e.g., `en`, `es`) or `None` for auto-detect |
| `batch_size` | integer | `64` | Batch size for processing |
| `timestamp` | string | `chunk` | `chunk` or `word` |
| `diarise_audio` | boolean | `false` | Enable speaker diarization |
| `webhook` | object | `null` | Webhook configuration |
| `is_async` | boolean | `false` | Run as background task |
| `managed_task_id` | string | `null` | Custom task ID |

#### POST `/audio/transcriptions` - OpenAI-Compatible

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | **required** | Audio file to transcribe |
| `model` | string | `whisper-large-v3-turbo` | Model name |
| `language` | string | `null` | Language code (optional) |
| `response_format` | string | `json` | Response format |

---

## 🔐 Authentication

If you set the `ADMIN_KEY` environment variable, all requests must include the `x-admin-api-key` header:

```bash
curl -X POST http://localhost:8000/ \
  -H "x-admin-api-key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

---

## 🌐 Open WebUI Integration

Configure Open WebUI to use this API for speech-to-text:

```bash
# Environment variables
export STT_ENGINE=openai
export STT_OPENAI_API_BASE_URL=http://localhost:8000
export STT_OPENAI_API_KEY=dummy
export STT_MODEL=whisper-large-v3-turbo

# Start Open WebUI
open-webui serve
```

### Docker Compose

```yaml
services:
  open-webui:
    environment:
      - STT_ENGINE=openai
      - STT_OPENAI_API_BASE_URL=http://host.docker.internal:8000
      - STT_OPENAI_API_KEY=dummy
      - STT_MODEL=whisper-large-v3-turbo
```

---

## ☁️ Deployment

### Fly.io (GPU)

```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Launch the app
fly launch

# Set secrets (optional)
fly secrets set ADMIN_KEY=your-token HF_TOKEN=your-hf-token

# Deploy
fly deploy
```

Your API will be available at: `https://your-app-name.fly.dev`

### Other Cloud Providers

This is a standard Docker application that can be deployed on:
- **AWS** (EC2 with GPU)
- **GCP** (Compute Engine with GPU)
- **Azure** (GPU VMs)
- **RunPod** (GPU instances)
- **Vast.ai** (GPU rentals)

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ADMIN_KEY` | No | API admin key for authentication |
| `HF_TOKEN` | No* | Hugging Face token for speaker diarization |

*Required only if using speaker diarization

### GPU Memory Configuration

For custom GPU memory limits, modify `app/app.py`:

```python
# Set memory fraction (e.g., 0.15 = 15%)
torch.cuda.set_per_process_memory_fraction(0.15, device=0)
```

### Model Configuration

The API uses **Whisper Large V3 Turbo** by default. To use a different model, modify `app/app.py`:

```python
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",  # Change model here
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"},
)
```

### Speaker Diarization Setup

To enable speaker diarization:

1. Accept user conditions:
   - [`pyannote/segmentation-3.0`](https://hf.co/pyannote/segmentation-3.0)
   - [`pyannote/speaker-diarization-3.1`](https://hf.co/pyannote/speaker-diarization-3.1)

2. Create access token at [`hf.co/settings/tokens`](https://hf.co/settings/tokens)

3. Set environment variable:
   ```bash
   export HF_TOKEN=your-token-here
   ```

---

## 📊 Performance Benchmarks

**Test Environment:** NVIDIA A100 80GB

| Optimization | Time for 150 min audio |
|--------------|------------------------|
| Large V3 Turbo + FP16 + Batch[24] + Flash Attn 2 | **~2 min (1:38)** |
| + Speaker Diarization | **~2 min (3:16)** |
| + Fly.io machine startup | **~2 min (1:58)** |
| + Diarization + Startup | **~2 min (3:36)** |

**Model Comparison:**

| Feature | Whisper V3 | Whisper V3 Turbo |
|---------|------------|------------------|
| Model Size | 3 GB | 1.62 GB |
| Speed | 1x | **2x faster** |
| Accuracy | Excellent | Excellent |
| Memory Usage | Higher | **Lower** |
| Languages | 99+ | 99+ |

---

## 🛠️ Troubleshooting

### API Not Starting

```bash
# Check if port is in use
lsof -i :8000

# Kill existing process
pkill -f "uvicorn app.app:app"

# Restart
uvicorn app.app:app --host 0.0.0.0 --port 8000
```

### Out of Memory Errors

- Reduce `batch_size` in the request
- Use a smaller model (e.g., `whisper-medium`)
- Increase GPU memory allocation
- Use Whisper V3 Turbo instead of regular V3

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Verify GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Connection Errors

- Verify API is running: `curl http://localhost:8000/`
- Check firewall settings
- For Docker: Use `host.docker.internal` instead of `localhost`

---

## 📁 Project Structure

```
insanely-fast-whisper-api/
├── app/
│   ├── app.py                    # Main FastAPI application
│   ├── diarization_pipeline.py   # Diarization logic
│   └── diarize.py                # Diarization utilities
├── Dockerfile                    # Docker configuration
├── fly.toml                      # Fly.io configuration
├── pyproject.toml                # Poetry dependencies
├── requirements.txt              # Pip dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🌟 Use Cases

- **Podcast Transcription** - Convert episodes to searchable text
- **Meeting Notes** - Transcribe calls and meetings
- **Content Creation** - Generate subtitles for videos
- **Accessibility** - Add captions to media
- **Voice Assistants** - Process voice commands
- **Research** - Analyze audio interviews
- **Education** - Transcribe lectures and courses

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Vaibhav Srivastav](https://github.com/Vaibhavs10) - Original [Insanely Fast Whisper CLI](https://github.com/Vaibhavs10/insanely-fast-whisper)
- [OpenAI](https://openai.com/) - Whisper model
- [Hugging Face](https://huggingface.co/) - Transformers library
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

## 🔗 Links

- **GitHub Repository:** [groxaxo/insanely-fast-whisper-api](https://github.com/groxaxo/insanely-fast-whisper-api)
- **Docker Hub:** [yoeven/insanely-fast-whisper-api](https://hub.docker.com/r/yoeven/insanely-fast-whisper-api)
- **Original CLI:** [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- **Fully Managed API:** [JigsawStack Speech-to-Text](https://jigsawstack.com/speech-to-text)

---

## 💼 Fully Managed API

Don't want to manage infrastructure? [JigsawStack](https://jigsawstack.com) provides a fully managed, scalable API with:

- ✅ **No Infrastructure** - We handle everything
- ✅ **Auto-scaling** - Pay per use
- ✅ **99.9% Uptime** - Production-ready
- ✅ **Global CDN** - Low latency worldwide
- ✅ **Free Tier** - Get started for free

**Sign up:** [jigsawstack.com](https://jigsawstack.com)

---

<div align="center">

**Made with ❤️ by the community**

⭐ Star us on GitHub — it motivates us a lot!

[Report Bug](https://github.com/groxaxo/insanely-fast-whisper-api/issues) · [Request Feature](https://github.com/groxaxo/insanely-fast-whisper-api/issues)

</div>
