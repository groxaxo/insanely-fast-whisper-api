# Open WebUI Integration Guide

## ✅ OpenAI-Compatible Endpoint Added

The API now includes an OpenAI-compatible endpoint at `/audio/transcriptions` that works seamlessly with Open WebUI.

## Configuration for Open WebUI

### Environment Variables

Set these in your Open WebUI environment:

```bash
# Speech-to-Text Configuration
STT_ENGINE=openai
STT_OPENAI_API_BASE_URL=http://localhost:8002
STT_OPENAI_API_KEY=dummy  # Can be any value, not validated
STT_MODEL=whisper-large-v3
```

### Alternative: Using .env file

If Open WebUI uses a `.env` file, add:

```env
STT_ENGINE=openai
STT_OPENAI_API_BASE_URL=http://localhost:8002
STT_OPENAI_API_KEY=dummy
STT_MODEL=whisper-large-v3
```

### Docker Compose Configuration

If running Open WebUI with Docker Compose, add to the environment section:

```yaml
services:
  open-webui:
    environment:
      - STT_ENGINE=openai
      - STT_OPENAI_API_BASE_URL=http://host.docker.internal:8002
      - STT_OPENAI_API_KEY=dummy
      - STT_MODEL=whisper-large-v3
```

**Note**: Use `http://host.docker.internal:8002` if Open WebUI is in Docker and the Whisper API is on the host machine.

## API Endpoint Details

### POST /audio/transcriptions

**OpenAI-compatible endpoint for audio transcription**

#### Request Format (multipart/form-data):
- `file` (required): Audio file to transcribe
- `model` (optional): Model name (default: "whisper-large-v3")
- `language` (optional): Language code (e.g., "en", "es", "fr")
- `response_format` (optional): Response format (default: "json")

#### Response Format:
```json
{
  "text": "Transcribed text here"
}
```

#### Example cURL:
```bash
curl -X POST http://localhost:8002/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-large-v3" \
  -F "language=en"
```

## Features

- ✅ OpenAI-compatible API format
- ✅ Automatic file cleanup after transcription
- ✅ Supports all common audio formats (mp3, wav, m4a, etc.)
- ✅ GPU-accelerated with Flash Attention 2
- ✅ Memory-limited to 15% GPU usage
- ✅ Reduced batch size (24) for stability
- ✅ Automatic language detection if not specified

## Troubleshooting

### Error: "Server Connection Error"

**Cause**: Open WebUI cannot reach the API endpoint.

**Solutions**:
1. Verify the API is running: `curl http://localhost:8002/audio/transcriptions`
2. Check firewall settings
3. If using Docker, ensure correct network configuration
4. Verify the URL in Open WebUI settings

### Error: "400 Bad Request"

**Cause**: Invalid audio file or format issue.

**Solutions**:
1. Ensure audio file is valid and not corrupted
2. Try converting to a common format (mp3, wav)
3. Check file size (API handles chunking automatically)

### Error: "500 Internal Server Error"

**Cause**: Processing error or GPU memory issue.

**Solutions**:
1. Check API logs: View the terminal where `start_gpu2_limited.sh` is running
2. Verify GPU 2 is available: `nvidia-smi`
3. Restart the API: `pkill -f "uvicorn app.app:app" && ./start_gpu2_limited.sh`

## Performance Notes

- **GPU Memory**: Limited to 15% (3.53 GB) on GPU 2
- **Batch Size**: Set to 24 for optimal performance with memory constraints
- **Processing Speed**: ~2 minutes for 150 minutes of audio (on A100)
- **Concurrent Requests**: Handled sequentially to maintain memory limits

## Testing the Integration

### 1. Test the API directly:
```bash
# Create a test audio file or use an existing one
curl -X POST http://localhost:8002/audio/transcriptions \
  -F "file=@test_audio.mp3" \
  -F "model=whisper-large-v3"
```

### 2. Test from Open WebUI:
1. Go to Open WebUI settings
2. Navigate to Audio settings
3. Configure STT settings as shown above
4. Try recording or uploading audio in a chat
5. The transcription should appear automatically

## API Status Check

```bash
# Check if API is running
curl http://localhost:8002/

# Should return: {"detail":"Method Not Allowed"}
# This confirms the API is running (GET not allowed on root)
```

## Restart Commands

```bash
# Stop the API
pkill -f "uvicorn app.app:app"

# Start the API
cd /home/op/CascadeProjects/insanely-fast-whisper-api
./start_gpu2_limited.sh
```

## Support

If you encounter issues:
1. Check the API logs in the terminal
2. Verify GPU status: `nvidia-smi`
3. Test the endpoint directly with cURL
4. Check Open WebUI logs for connection errors
