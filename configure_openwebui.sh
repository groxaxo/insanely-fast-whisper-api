#!/bin/bash

# Script to configure Open WebUI to use the local Whisper API

echo "================================"
echo "Open WebUI Configuration Helper"
echo "================================"
echo ""

# Check if Open WebUI is running
if pgrep -f "open-webui" > /dev/null; then
    echo "✅ Open WebUI is currently running (PID: $(pgrep -f 'open-webui'))"
    echo ""
    echo "⚠️  You need to restart Open WebUI with the following environment variables:"
else
    echo "❌ Open WebUI is not running"
    echo ""
    echo "Start Open WebUI with the following environment variables:"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Environment Variables:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
cat << 'EOF'
export STT_ENGINE=openai
export STT_OPENAI_API_BASE_URL=http://localhost:8002
export STT_OPENAI_API_KEY=dummy
export STT_MODEL=whisper-large-v3
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Restart Commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Option 1: Restart with environment variables"
echo "─────────────────────────────────────────────"
cat << 'EOF'
# Stop Open WebUI
pkill -f "open-webui"

# Start with STT configuration
STT_ENGINE=openai \
STT_OPENAI_API_BASE_URL=http://localhost:8002 \
STT_OPENAI_API_KEY=dummy \
STT_MODEL=whisper-large-v3 \
open-webui serve
EOF

echo ""
echo "Option 2: Add to your shell profile (~/.bashrc or ~/.zshrc)"
echo "─────────────────────────────────────────────────────────────"
cat << 'EOF'
export STT_ENGINE=openai
export STT_OPENAI_API_BASE_URL=http://localhost:8002
export STT_OPENAI_API_KEY=dummy
export STT_MODEL=whisper-large-v3
EOF

echo ""
echo "Then restart Open WebUI: pkill -f 'open-webui' && open-webui serve"
echo ""

echo "Option 3: If using Docker"
echo "─────────────────────────"
cat << 'EOF'
docker run -d \
  -p 3000:8080 \
  -e STT_ENGINE=openai \
  -e STT_OPENAI_API_BASE_URL=http://host.docker.internal:8002 \
  -e STT_OPENAI_API_KEY=dummy \
  -e STT_MODEL=whisper-large-v3 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing the Integration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Restart Open WebUI with the environment variables above"
echo "2. Open Open WebUI in your browser"
echo "3. Go to Settings → Audio"
echo "4. Try recording or uploading audio in a chat"
echo "5. The transcription should appear automatically"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Troubleshooting:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If you see 'Server Connection Error':"
echo "  • Verify Whisper API is running: curl http://localhost:8002/"
echo "  • Check Open WebUI logs for connection errors"
echo "  • If using Docker, ensure network connectivity"
echo ""
echo "If transcription fails:"
echo "  • Check Whisper API logs in the terminal"
echo "  • Verify GPU is available: nvidia-smi"
echo "  • Try restarting the Whisper API"
echo ""
