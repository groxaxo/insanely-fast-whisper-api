#!/bin/bash

# Test script for Open WebUI integration
# This script tests the /audio/transcriptions endpoint

echo "================================"
echo "Testing Whisper API for Open WebUI"
echo "================================"
echo ""

# Check if API is running
echo "1. Checking if API is running..."
if curl -s -f http://localhost:8002/ > /dev/null 2>&1; then
    echo "   ❌ API returned unexpected success on GET /"
elif curl -s http://localhost:8002/ 2>&1 | grep -q "Method Not Allowed"; then
    echo "   ✅ API is running on port 8002"
else
    echo "   ❌ API is not responding. Please start it with: ./start_gpu2_limited.sh"
    exit 1
fi

echo ""
echo "2. Testing /audio/transcriptions endpoint..."

# Create a minimal test (this will fail but shows the endpoint is working)
RESPONSE=$(curl -s -X POST http://localhost:8002/audio/transcriptions \
    -F "file=@/dev/null" \
    -F "model=whisper-large-v3" 2>&1)

if echo "$RESPONSE" | grep -q "Soundfile is either not in the correct format"; then
    echo "   ✅ Endpoint is responding correctly (expected error for invalid file)"
elif echo "$RESPONSE" | grep -q "text"; then
    echo "   ✅ Endpoint is working! Response: $RESPONSE"
else
    echo "   ❌ Unexpected response: $RESPONSE"
    exit 1
fi

echo ""
echo "================================"
echo "✅ API is ready for Open WebUI!"
echo "================================"
echo ""
echo "Configure Open WebUI with these settings:"
echo ""
echo "  STT_ENGINE=openai"
echo "  STT_OPENAI_API_BASE_URL=http://localhost:8002"
echo "  STT_OPENAI_API_KEY=dummy"
echo "  STT_MODEL=whisper-large-v3"
echo ""
echo "If Open WebUI is in Docker, use:"
echo "  STT_OPENAI_API_BASE_URL=http://host.docker.internal:8002"
echo ""
echo "To set these in Open WebUI:"
echo "1. Restart Open WebUI with environment variables, OR"
echo "2. Set them in Open WebUI's admin settings (if available)"
echo ""
