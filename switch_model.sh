#!/bin/bash
# Quick model switcher for NPGlue

echo "🤖 NPGlue Model Switcher"
echo "========================"

# Check available models
echo "📂 Available models:"
if [ -d "models/qwen3-8b-int8" ]; then
    echo "   1) Qwen3-8B-INT8 (~6-8GB memory, best quality)"
fi
if [ -d "models/qwen3-0.6b-fp16" ]; then
    echo "   2) Qwen3-0.6B-FP16 (~1-2GB memory, fast)"
fi

echo
echo "💡 Current setting:"
if [ -f ".model_config" ]; then
    cat .model_config
else
    echo "   No model configured"
fi

echo
read -p "Choose model (1 or 2): " choice

case $choice in
    1)
        if [ -d "models/qwen3-8b-int8" ]; then
            echo "MODEL_PATH=models/qwen3-8b-int8" > .model_config
            echo "✅ Switched to Qwen3-8B-INT8 (high quality, more memory)"
        else
            echo "❌ Qwen3-8B-INT8 not found. Run ./install to download it."
        fi
        ;;
    2)
        if [ -d "models/qwen3-0.6b-fp16" ]; then
            echo "MODEL_PATH=models/qwen3-0.6b-fp16" > .model_config
            echo "✅ Switched to Qwen3-0.6B-FP16 (fast, less memory)"
        else
            echo "❌ Qwen3-0.6B-FP16 not found. Run ./install to download it."
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo
echo "🔄 Restart your NPGlue server for the change to take effect:"
echo "   ./start_server.sh"
