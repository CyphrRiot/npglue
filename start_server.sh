#!/bin/bash
# Simple server startup script

cd "$(dirname "$0")"

if [ ! -d "openvino-env" ]; then
    echo "❌ Virtual environment not found. Run ./install.sh first"
    exit 1
fi

if [ ! -d "models/deepseek-r1-fp16-ov" ]; then
    echo "❌ Model not found. Run ./install.sh first"
    exit 1
fi

echo "🚀 Starting DeepSeek-R1 FP16 OpenVINO Server..."
echo "📖 API docs: http://localhost:8000/docs"
echo "🔍 Health: http://localhost:8000/health"
echo "💡 Use Ctrl+C to stop"

source openvino-env/bin/activate
python server_production.py
