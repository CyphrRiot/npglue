# ɴᴘɢʟᴜᴇ - Intel NPU Ollama Replacement!

**ɴᴘɢʟᴜᴇ** provides a complete setup for running **multiple AI models** locally using OpenVINO for AI-assisted coding and development with **direct, quality answers**.

![I am a Pickle](Image/pickle_rick.png)

## 🚀 **Quick Start**

```bash
git clone https://github.com/CyphrRiot/npglue.git
cd npglue
./install
```

The installer will ask you to choose your model from **8 options**:

### **OpenVINO Pre-Optimized Models (fastest setup):**
- **Qwen3-8B-INT8** (~6-8GB) - Best quality for complex tasks
- **Qwen3-0.6B-FP16** (~1-2GB) - Fast and lightweight  
- **OpenLlama-7B-INT4** (~4-5GB) - Great balance for coding
- **OpenLlama-3B-INT4** (~2-3GB) - Lightweight with good performance

### **Community Pre-Optimized Models:**
- **Llama-3.1-8B-INT4** (~5-6GB) - Latest Llama with excellent coding abilities

### **Convert-on-Install Models:**
- **Phi-3-Mini-4K** (~4GB) - Microsoft model optimized for NPU
- **DeepSeek-Coder-6.7B** (~6-7GB) - Specialized coding model, excellent for development  
- **DeepSeek-Coder-1.3B** (~2GB) - Lightweight coding specialist

## ✅ **What You Get**

- **Multiple AI Models**: 8 models to choose from - Qwen3, Llama, Phi-3, DeepSeek coding specialists
- **Model Choice**: Pick based on your needs - quality vs speed vs coding specialization
- **Easy Model Switching**: Use `./switch_model.sh` to change models anytime (no reinstall needed!)  
- **OpenVINO Optimized**: Fast inference optimized for Intel NPU/GPU hardware
- **20-30+ tokens/sec**: Fast local inference with memory efficiency
- **Performance Display**: Every response shows completion time and token rate
- **Direct Answers**: No rambling - get concise, actionable responses
- **Zed Compatible**: Works as Ollama provider (no API key hassles!)
- **Full Ollama API**: Complete compatibility with Ollama ecosystem
- **Dual API Support**: Both OpenAI and Ollama compatible endpoints
- **Goose Ready**: Drop-in replacement for OpenAI API

## 🔧 **Requirements**

- **OS**: Linux (Arch/CachyOS recommended)
- **Memory**: 2GB+ RAM (for 0.6B model) or 8GB+ RAM (for 8B model)
- **Storage**: 10-15GB free space
- **CPU**: Intel preferred (excellent OpenVINO optimization)
- **Shell**: Compatible with bash, zsh, and fish
- **Hardware acceleration**:
    - **Best**: Intel NPU (12th gen+ processors) - 20-30 tokens/sec
    - **Good**: Intel integrated GPU - 5-10 tokens/sec
    - **Basic**: Any CPU - 2-5 tokens/sec (slower but functional)

## 📊 **Performance Monitoring**

ɴᴘɢʟᴜᴇ automatically displays performance metrics with every response:

```
What is the capital of France?

The capital of France is Paris.

*Completed in 0.85 seconds at 23.2 tokens/sec*
```

**Benefits:**

- **Real-time feedback** on AI response speed
- **Performance monitoring** under different loads
- **Model comparison** when testing different configurations
- **System optimization** insights for tuning

This helps you:

- Monitor system performance
- Compare model variants (8B vs 0.6B)
- Identify when your system needs optimization
- Debug slow response issues

**Token Limits:**

- **Respects user preferences** - Request up to 4096 tokens
- **No artificial caps** - Let the model complete naturally
- **Smart defaults** - 200 tokens if not specified
- **Memory aware** - Monitors available RAM during generation

## 🛠️ **Performance Optimization**

ɴᴘɢʟᴜᴇ includes built-in tools to diagnose and optimize performance:

### **Model Switching**

```bash
./switch_model.sh
```

Easily switch between models based on your needs:

- **8B model**: Maximum quality for complex tasks (needs 8GB+ RAM)
- **0.6B model**: Speed and efficiency for quick responses (needs 2GB+ RAM)

**Tip**: If you're getting slow performance (under 15 tok/sec), run the diagnostics tool to identify memory pressure or other issues.

### **CPU Performance Management**

```bash
# Manual CPU optimization
./boost_cpu.sh           # Set CPU to performance mode

# Manual CPU restoration
./restore_cpu.sh         # Restore power-saving mode

# Automatic management (recommended)
./start_server.sh        # Auto-saves/restores CPU settings
```

**Note**: `start_server.sh` automatically saves your CPU governor settings and restores them when you press `Ctrl+C` or exit the server.

## 📊 **Performance Expectations**

| Model                     | Size   | Memory   | Speed (NPU) | Speed (iGPU) | Speed (CPU) | Best For                              |
| ------------------------- | ------ | -------- | ----------- | ------------ | ----------- | ------------------------------------- |
| **Qwen3-8B-INT8**         | ~6-8GB | 8GB+ RAM | 20-30 tok/s | 5-10 tok/s   | 2-5 tok/s   | Complex tasks, detailed explanations |
| **Qwen3-0.6B-FP16**       | ~1-2GB | 2GB+ RAM | 25-40 tok/s | 8-15 tok/s   | 4-8 tok/s   | Quick answers, simple tasks           |
| **OpenLlama-7B-INT4**     | ~4-5GB | 6GB+ RAM | 22-35 tok/s | 6-12 tok/s   | 3-6 tok/s   | Balanced coding and general tasks     |
| **OpenLlama-3B-INT4**     | ~2-3GB | 4GB+ RAM | 30-45 tok/s | 10-18 tok/s  | 5-9 tok/s   | Fast responses, lightweight           |
| **Llama-3.1-8B-INT4**     | ~5-6GB | 8GB+ RAM | 20-30 tok/s | 5-10 tok/s   | 2-5 tok/s   | Latest Llama, excellent coding        |
| **Phi-3-Mini-4K**         | ~4GB   | 6GB+ RAM | 25-35 tok/s | 7-14 tok/s   | 3-7 tok/s   | NPU-optimized, Microsoft quality      |
| **DeepSeek-Coder-6.7B**   | ~6-7GB | 8GB+ RAM | 18-28 tok/s | 4-9 tok/s    | 2-4 tok/s   | Best for coding, development tasks    |
| **DeepSeek-Coder-1.3B**   | ~2GB   | 3GB+ RAM | 35-50 tok/s | 12-20 tok/s  | 6-10 tok/s  | Fast coding assistant, lightweight    |

## 🛠️ **What the Installer Does**

### **System Setup:**

- ✅ Checks system requirements (RAM, disk space)
- ✅ Installs system dependencies (Python, OpenVINO drivers, etc.)
- ✅ Creates clean Python virtual environment
- ✅ Installs **CPU-only** AI packages (OpenVINO 2024.x, transformers, PyTorch-CPU)

### **Model Setup:**

- ✅ **Interactive model choice**: Pick Qwen3-8B-INT8 or Qwen3-0.6B-FP16
- ✅ Downloads your chosen optimized OpenVINO model
- ✅ Memory-safe verification (no crashes during setup)
- ✅ CPU performance optimization

### **Configuration Instructions:**

- ✅ **Safe Goose setup**: Checks for existing config, won't overwrite
- ✅ **Zed integration**: Exact settings for assistant configuration
- ✅ **Testing steps**: How to verify everything works properly

## 🦆 **Goose Integration (Safe)**

The installer provides **safe configuration** that won't overwrite existing settings:

**If you DON'T have Goose configured:**

```bash
mkdir -p ~/.config/goose
cp goose_config_example.yaml ~/.config/goose/config.yaml
# No API key needed! Uses Ollama provider which is simpler.
```

**If you HAVE existing Goose config, just add:**

```yaml
GOOSE_PROVIDER: ollama
GOOSE_MODEL: qwen3
OLLAMA_HOST: http://localhost:11434
```

**Why Ollama provider?** ɴᴘɢʟᴜᴇ supports both OpenAI and Ollama APIs, but Goose's Ollama provider doesn't require API key setup - much simpler!

## ⚡ **Zed Integration (WORKING!)**

**ɴᴘɢʟᴜᴇi works as an Ollama provider** (no API key hassles!):

```json
{
    "language_models": {
        "ollama": {
            "api_url": "http://localhost:11434",
            "available_models": [
                {
                    "name": "qwen3",
                    "display_name": "Qwen3 Local",
                    "max_tokens": 4096,
                    "supports_tools": true
                }
            ]
        }
    },
    "agent": {
        "default_model": {
            "provider": "ollama",
            "model": "qwen3"
        }
    }
}
```

**Why this works:** Zed's OpenAI provider is finicky about API keys, but the Ollama provider "just works"!

## 🧪 **Testing Your Installation**

After running `./install`, test with:

```bash
# Start the server
./start_server.sh

# Test health
curl http://localhost:11434/health

# Test Ollama API (for Zed)
curl http://localhost:11434/api/tags

# Test OpenAI API (for Goose)
curl http://localhost:11434/v1/models

# Run full model test
```

## 🔌 **API Endpoints**

ɴᴘɢʟᴜᴇ provides **complete API compatibility** with both OpenAI and Ollama:

**OpenAI API** (for Goose):

- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions
- `GET /health` - Health check

**Ollama API** (for Zed):

- `GET /api/tags` - List models
- `POST /api/chat` - Chat completions
- `POST /api/generate` - Text generation
- `POST /api/show` - Model details
- `GET /api/version` - Version info
- `POST /api/pull` - Model management (returns success for local models)

**Utilities:**

- `GET /models` - System information
- `POST /unload` - Unload model from memory

## 🌍 **Works With**

- **Goose**: AI development assistant
- **Zed**: Modern code editor
- **Cursor**: AI-powered IDE
- **Continue.dev**: VS Code extension
- **Any OpenAI-compatible client**

## 📁 **Project Structure**

```
npglue/
├── install                    # 🌟 Beautiful one-command installer
├── start_server.sh            # Start the AI server (auto CPU cleanup on exit)
├── server_production.py       # FastAPI server with dual API compatibility
├── boost_cpu.sh               # CPU performance optimization
├── restore_cpu.sh             # 🔄 Restore CPU to power-saving mode
├── switch_model.sh            # 🔄 Easy model switching utility
├── goose_config_example.yaml  # Safe Goose configuration template
├── README.md                  # This documentation
├── LICENSE                    # License file
└── models/                    # Downloaded models (created by installer)
    ├── qwen3-8b-int8/         # High quality model (8GB)
    └── qwen3-0.6b-fp16/       # Fast model (1-2GB)
```

## 🎯 **Why Choose ɴᴘɢʟᴜᴇ?**

- **One Command Setup**: `./install` does everything beautifully
- **Model Choice**: Choose between quality (8B) or speed (0.6B)
- **Memory Safe**: Won't crash during installation or use
- **Configuration Safe**: Won't overwrite your existing tool settings
- **Expert Optimized**: Uses official OpenVINO optimized models
- **Direct Answers**: No rambling - designed for practical Q&A
- **Clear Instructions**: Tells you exactly what to do next
- **Local Privacy**: No data sent to external APIs
- **Fast Performance**: Optimized for Intel hardware
- **Production Ready**: Proper error handling and monitoring

## 🔧 **Advanced Usage**

### **API Endpoints:**

- **Chat**: `http://localhost:11434/v1/chat/completions` (OpenAI compatible)
- **Health**: `http://localhost:11434/health`
- **Docs**: `http://localhost:11434/docs`

### **Environment Control:**

```bash
# Activate environment manually
source npglue-env/bin/activate

# Check available devices
python -c "import openvino; print(openvino.Core().available_devices)"
```

## 🚀 **Recent Improvements**

- ✅ **Complete Ollama API**: Added `/api/show`, `/api/version`, `/api/pull` endpoints (no more 404s!)
- ✅ **Model Switching**: Easy utility to switch between 8B and 0.6B models (`switch_model.sh`)
- ✅ **Memory Optimization**: Automatic detection and fixes for memory pressure issues
- ✅ **Flexible Token Limits**: Respects user preferences up to 4096 tokens (no more artificial caps!)
- ✅ **Performance Display**: All responses now show "Completed in X.XX seconds at X.X tokens/sec"
- ✅ **Model Choice Menu**: Pick Qwen3-8B-INT8 OR Qwen3-0.6B-FP16 during install
- ✅ **Switched from DeepSeek-R1**: No more rambling - direct answers now!
- ✅ **CPU-Only Install**: No NVIDIA dependencies on Intel systems
- ✅ **Dual API Support**: Both OpenAI AND Ollama compatible endpoints
- ✅ **Zed Integration Fixed**: Works as Ollama provider (no API key issues!)
- ✅ **Safe configuration**: Protects existing Goose/Zed settings
- ✅ **Simplified installer**: One beautiful command does everything
- ✅ **Flexible memory**: 2GB+ (0.6B) or 8GB+ (8B) requirements
- ✅ **Expert models**: Official OpenVINO optimized versions

---

**ɴᴘɢʟᴜᴇ: One command to local AI coding bliss!** 🚀

_Get the power of Qwen3's direct, practical responses running locally on your machine in minutes._
