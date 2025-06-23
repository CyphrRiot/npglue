#!/bin/bash
# Enhanced model switcher for NPGlue

echo "🤖 NPGlue Enhanced Model Switcher"
echo "=================================="

# Check available models
echo "📂 Available models:"
available_models=()
model_choices=()

if [ -d "models/qwen3-8b-int8" ]; then
    available_models+=("Qwen3-8B-INT8 (~6-8GB memory, best quality)")
    model_choices+=("models/qwen3-8b-int8|qwen3-8b")
    echo "   1) Qwen3-8B-INT8 (~6-8GB memory, best quality)"
fi

if [ -d "models/qwen3-0.6b-fp16" ]; then
    available_models+=("Qwen3-0.6B-FP16 (~1-2GB memory, fast)")
    model_choices+=("models/qwen3-0.6b-fp16|qwen3-0.6b")
    echo "   2) Qwen3-0.6B-FP16 (~1-2GB memory, fast)"
fi

if [ -d "models/open-llama-7b-int4" ]; then
    available_models+=("OpenLlama-7B-INT4 (~4-5GB memory, great balance)")
    model_choices+=("models/open-llama-7b-int4|open-llama-7b")
    echo "   3) OpenLlama-7B-INT4 (~4-5GB memory, great balance)"
fi

if [ -d "models/open-llama-3b-int4" ]; then
    available_models+=("OpenLlama-3B-INT4 (~2-3GB memory, lightweight)")
    model_choices+=("models/open-llama-3b-int4|open-llama-3b")
    echo "   4) OpenLlama-3B-INT4 (~2-3GB memory, lightweight)"
fi

if [ -d "models/llama-3.1-8b-int4" ]; then
    available_models+=("Llama-3.1-8B-INT4 (~5-6GB memory, excellent coding)")
    model_choices+=("models/llama-3.1-8b-int4|llama-3.1-8b")
    echo "   5) Llama-3.1-8B-INT4 (~5-6GB memory, excellent coding)"
fi

if [ -d "models/phi-3-mini-4k" ]; then
    available_models+=("Phi-3-Mini-4K (~4GB memory, optimized for NPU)")
    model_choices+=("models/phi-3-mini-4k|phi-3-mini")
    echo "   6) Phi-3-Mini-4K (~4GB memory, optimized for NPU)"
fi

if [ -d "models/deepseek-coder-6.7b" ]; then
    available_models+=("DeepSeek-Coder-6.7B (~6-7GB memory, coding specialist)")
    model_choices+=("models/deepseek-coder-6.7b|deepseek-coder-6.7b")
    echo "   7) DeepSeek-Coder-6.7B (~6-7GB memory, coding specialist)"
fi

if [ -d "models/deepseek-coder-1.3b" ]; then
    available_models+=("DeepSeek-Coder-1.3B (~2GB memory, light coding specialist)")
    model_choices+=("models/deepseek-coder-1.3b|deepseek-coder-1.3b")
    echo "   8) DeepSeek-Coder-1.3B (~2GB memory, light coding specialist)"
fi

if [ ${#available_models[@]} -eq 0 ]; then
    echo "❌ No models found. Run ./install to download models."
    exit 1
fi

echo
echo "💡 Current setting:"
if [ -f ".model_config" ]; then
    cat .model_config
else
    echo "   No model configured"
fi

echo
read -p "Choose model (1-${#available_models[@]}): " choice

# Validate choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#available_models[@]} ]; then
    echo "❌ Invalid choice. Please enter a number between 1 and ${#available_models[@]}"
    exit 1
fi

# Get the selected model info
selected_index=$((choice - 1))
selected_model_info="${model_choices[$selected_index]}"
model_path=$(echo "$selected_model_info" | cut -d'|' -f1)
model_id=$(echo "$selected_model_info" | cut -d'|' -f2)

if [ -d "$model_path" ]; then
    echo "MODEL_PATH=$model_path" > .model_config
    echo "MODEL_ID=$model_id" >> .model_config
    echo "✅ Switched to ${available_models[$selected_index]}"
    echo "📝 Model ID: $model_id"
else
    echo "❌ Model directory not found: $model_path"
    echo "   Run ./install to download this model."
    exit 1
fi

echo
echo "🔄 Restart your NPGlue server for the change to take effect:"
echo "   ./start_server.sh"
