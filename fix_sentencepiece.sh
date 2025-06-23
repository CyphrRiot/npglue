#!/bin/bash
# Fix SentencePiece installation for OpenLlama models

echo "🔧 NPGlue SentencePiece Installer"
echo "================================"
echo "This script helps install SentencePiece for OpenLlama models"
echo

cd "$(dirname "$0")"
source npglue-env/bin/activate

echo "📦 Installing build dependencies..."
sudo pacman -S --needed --noconfirm base-devel cmake

echo "🔧 Attempting SentencePiece installation..."

# Try different methods
if pip install sentencepiece; then
    echo "✅ SentencePiece installed successfully!"
else
    echo "❌ SentencePiece build failed"
    echo
    echo "💡 Alternative approaches:"
    echo "1. Use Qwen3, Phi-3, or DeepSeek models (don't need SentencePiece)"
    echo "2. Try installing from conda-forge:"
    echo "   conda install -c conda-forge sentencepiece"
    echo
    echo "3. Or manually build SentencePiece:"
    echo "   git clone https://github.com/google/sentencepiece.git"
    echo "   cd sentencepiece && mkdir build && cd build"
    echo "   cmake .. && make -j && sudo make install"
    echo
    echo "🔄 Switch to a working model:"
    echo "./switch_model.sh"
    exit 1
fi

echo
echo "✅ Ready to use OpenLlama models!"
echo "🔄 Switch to an OpenLlama model:"
echo "./switch_model.sh"
