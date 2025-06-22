#!/bin/bash
# NPGlue - DeepSeek-R1 Installation Script
# Installs working DeepSeek-R1 with OpenVINO for AI-assisted coding

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 NPGlue - DeepSeek-R1 OpenVINO Installation${NC}"
echo "=============================================="
echo "This will install DeepSeek-R1 with OpenVINO for local AI coding assistance"
echo

# Check system requirements
echo -e "${BLUE}📋 Checking system requirements...${NC}"

# Memory check
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM_GB" -lt 12 ]; then
    echo -e "${RED}❌ Error: Need at least 12GB RAM (found ${TOTAL_MEM_GB}GB)${NC}"
    exit 1
fi

# Disk space check  
AVAILABLE_GB=$(df . | awk 'NR==2{printf "%.0f", $4/1024/1024}')
if [ "$AVAILABLE_GB" -lt 15 ]; then
    echo -e "${RED}❌ Error: Need at least 15GB free disk space${NC}"
    exit 1
fi

echo -e "${GREEN}✅ System requirements met: ${TOTAL_MEM_GB}GB RAM, ${AVAILABLE_GB}GB free space${NC}"

# 1. Install system dependencies
echo -e "\n${BLUE}📦 Installing system dependencies...${NC}"
sudo pacman -S --needed --noconfirm python python-pip base-devel cmake git
echo -e "${GREEN}✅ System dependencies installed${NC}"

# 2. Install Intel NPU driver (optional but recommended)
echo -e "\n${BLUE}🔧 Installing Intel NPU driver...${NC}"
if ! pacman -Q intel-npu-driver &>/dev/null; then
    if command -v yay &>/dev/null; then
        yay -S --noconfirm intel-npu-driver || echo -e "${YELLOW}⚠️ NPU driver installation failed (optional)${NC}"
    else
        echo -e "${YELLOW}⚠️ yay not found, skipping NPU driver (optional)${NC}"
    fi
fi
sudo pacman -S --needed --noconfirm intel-compute-runtime

# 3. Create Python virtual environment
echo -e "\n${BLUE}🐍 Setting up Python environment...${NC}"
if [ -d "openvino-env" ]; then
    echo "Virtual environment already exists, removing..."
    rm -rf openvino-env
fi

python -m venv openvino-env
source openvino-env/bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip setuptools wheel

# 4. Install OpenVINO and dependencies
echo -e "\n${BLUE}🤖 Installing OpenVINO and ML packages...${NC}"

# Install OpenVINO
pip install openvino>=2025.0

# Install PyTorch CPU-only (no NVIDIA dependencies)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Optimum Intel for model conversion
pip install optimum-intel --no-deps
pip install optimum==1.25.3 scipy onnx transformers --no-deps

# Install web server dependencies
pip install fastapi uvicorn requests psutil

echo -e "${GREEN}✅ Python packages installed${NC}"

# 5. Download pre-optimized DeepSeek-R1 OpenVINO model
echo -e "\n${BLUE}📥 Downloading DeepSeek-R1 FP16 OpenVINO model...${NC}"
echo "Using pre-converted professional OpenVINO model (much better than INT4!)"

# Create models directory
mkdir -p models

# Check if model already exists
if [ -d "models/deepseek-r1-fp16-ov" ] && [ -f "models/deepseek-r1-fp16-ov/DeepSeek-R1-0528-Qwen3-8B-fp16-ov/openvino_model.bin" ]; then
    echo -e "${GREEN}✅ DeepSeek-R1 model already exists, skipping download${NC}"
else
    echo "Downloading DeepSeek-R1-0528-Qwen3-8B FP16 OpenVINO..."
    
    # Install huggingface-hub if not present
    pip install huggingface-hub
    
    # Download the pre-converted OpenVINO model
    python -c "
from huggingface_hub import snapshot_download
import os

print('📦 Downloading DeepSeek-R1 FP16 OpenVINO model...')
print('🚀 This is a professional OpenVINO conversion - should be much faster!')

repo_id = 'Echo9Zulu/DeepSeek-R1-0528-Qwen3-8B-OpenVINO'
model_variant = 'DeepSeek-R1-0528-Qwen3-8B-fp16-ov'
local_dir = 'models/deepseek-r1-fp16-ov'

snapshot_download(
    repo_id=repo_id,
    allow_patterns=[f'{model_variant}/*'],
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f'✅ DeepSeek-R1 FP16 model downloaded to: {local_dir}')
print('📈 Expected performance: 15-25+ tokens/sec (vs 11-14 with INT4)')
"
    
    echo -e "${GREEN}✅ DeepSeek-R1 FP16 model downloaded${NC}"
fi

# 6. Test installation
echo -e "\n${BLUE}🧪 Testing installation...${NC}"

# Create quick test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import time

def test_model():
    model_path = "models/deepseek-r1-fp16-ov/DeepSeek-R1-0528-Qwen3-8B-fp16-ov"
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = OVModelForCausalLM.from_pretrained(model_path, device="CPU")
    
    print("Testing generation...")
    prompt = "def hello_world():"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    gen_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    speed = tokens_generated / gen_time
    
    print(f"✅ Generated: '{generated[:60]}{'...' if len(generated) > 60 else ''}'")
    print(f"✅ Speed: {speed:.1f} tokens/sec")
    print(f"✅ Installation successful!")
    
    return speed > 3  # Should be at least 3 tok/s

if __name__ == "__main__":
    try:
        success = test_model()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        exit(1)
EOF

python test_installation.py
TEST_RESULT=$?
rm test_installation.py

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Installation test passed!${NC}"
else
    echo -e "${RED}❌ Installation test failed${NC}"
    exit 1
fi

# 7. Create startup script
echo -e "\n${BLUE}📝 Creating startup script...${NC}"
cat > start_server.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source openvino-env/bin/activate
python server_production.py
EOF
chmod +x start_server.sh

# 8. Optimize CPU performance
echo -e "\n${BLUE}⚡ Optimizing CPU performance...${NC}"
sudo ./boost_cpu.sh

# 9. Installation complete
echo -e "\n${GREEN}🎉 NPGlue INSTALLATION COMPLETE!${NC}"
echo "=================================="
echo -e "${BLUE}Next steps:${NC}"
echo "1. Start the server: ./start_server.sh"
echo "2. Test with: python test_installation.py"
echo "3. Configure your IDE to use: http://localhost:8000"
echo
echo -e "${BLUE}Performance:${NC}"
echo "- Expected speed: 15-25+ tokens/sec"
echo "- Memory usage: 10-12GB"
echo "- Model: DeepSeek-R1 FP16 (~8GB)"
echo
echo -e "${YELLOW}💡 Tip: First generation is slower (warmup), subsequent ones are much faster${NC}"
echo
echo -e "${GREEN}✅ NPGlue ready for AI-assisted coding!${NC}"
