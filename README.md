# NPGlue - Intel NPU Glue for AI

**NPGlue** provides a complete setup for running **Qwen3** models locally using OpenVINO for AI-assisted coding and development with **direct, quality answers**.

## 🚀 **Quick Start**

```bash
git clone https://github.com/CyphrRiot/npglue.git
cd npglue
./install
```

The installer will ask you to choose your model:
- **Qwen3-8B-INT8** (~6-8GB) - Best quality for complex tasks
- **Qwen3-0.6B-FP16** (~1-2GB) - Fast and lightweight

## ✅ **What You Get**

- **Qwen3 Models**: Purpose-built for direct Q&A and coding assistance
- **Model Choice**: Pick 8B for quality or 0.6B for speed
- **OpenVINO Optimized**: Fast inference optimized for Intel hardware
- **20-50+ tokens/sec**: Fast local inference with memory efficiency
- **Direct Answers**: No more rambling - get "4" when you ask "What is 2+2"
- **Zed Compatible**: Works as Ollama provider (no API key hassles!)
- **Dual API Support**: Both OpenAI and Ollama compatible endpoints
- **Goose Ready**: Drop-in replacement for OpenAI API

## 🔧 **Requirements**

- **OS**: Linux (Arch/CachyOS recommended)  
- **Memory**: 2GB+ RAM (for 0.6B model) or 8GB+ RAM (for 8B model)
- **Storage**: 10-15GB free space  
- **CPU**: Intel preferred (excellent OpenVINO optimization)
- **Optional**: Intel NPU for potential acceleration

## 📊 **Performance Expectations**

| Model | Size | Memory | Speed | Quality | Best For |
|-------|------|---------|-------|---------|-----------|
| **Qwen3-8B-INT8** | ~6-8GB | 8GB+ RAM | 20-30 tok/s | Excellent | Complex coding, detailed explanations |
| **Qwen3-0.6B-FP16** | ~1-2GB | 2GB+ RAM | 40-60 tok/s | Good | Quick answers, simple tasks |

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
```

**If you HAVE existing Goose config, just add:**
```yaml
provider: openai
model: qwen3-openvino  
api_base: http://localhost:8000/v1
api_key: local-key
```

## ⚡ **Zed Integration (WORKING!)**

**NPGlue works as an Ollama provider** (no API key hassles!):

```json
{
  "language_models": {
    "ollama": {
      "api_url": "http://localhost:8000",
      "available_models": [
        {
          "name": "qwen3-openvino",
          "display_name": "Qwen3 Local",
          "max_tokens": 32768,
          "supports_tools": true
        }
      ]
    }
  },
  "agent": {
    "default_model": {
      "provider": "ollama",
      "model": "qwen3-openvino"
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
curl http://localhost:8000/health

# Test Ollama API (for Zed)
curl http://localhost:8000/api/tags

# Test OpenAI API (for Goose)  
curl http://localhost:8000/v1/models

# Run full model test  
python test_installation.py
```

## 🔌 **API Endpoints**

NPGlue provides **dual API compatibility**:

**Ollama API** (for Zed):
- `GET /api/tags` - List models
- `POST /api/chat` - Chat completions  
- `POST /api/generate` - Text generation

**OpenAI API** (for Goose):
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions
- `GET /health` - Health check

## 🌍 **Works With**

- **Goose**: AI development assistant
- **Zed**: Modern code editor  
- **Cursor**: AI-powered IDE
- **Continue.dev**: VS Code extension
- **Any OpenAI-compatible client**

## 📁 **Project Structure**

```
npglue/
├── install                     # 🌟 Beautiful one-command installer
├── start_server.sh             # Start the AI server  
├── server_production.py        # FastAPI server with OpenAI API
├── test_installation.py        # Verify installation works
├── boost_cpu.sh               # CPU performance optimization
├── goose_config_example.yaml  # Safe Goose configuration template
├── README.md                  # This documentation
├── npglue-env/               # Python environment (created)
├── models/                    # Your chosen model (downloaded)
    ├── qwen3-8b-int8/         # OR
    └── qwen3-0.6b-fp16/       # Depending on your choice
```

## 🎯 **Why Choose NPGlue?**

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
- **Chat**: `http://localhost:8000/v1/chat/completions` (OpenAI compatible)
- **Health**: `http://localhost:8000/health`
- **Docs**: `http://localhost:8000/docs`

### **Environment Control:**
```bash
# Activate environment manually
source npglue-env/bin/activate

# Check available devices
python -c "import openvino; print(openvino.Core().available_devices)"
```

## 🚀 **Recent Improvements**

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

**NPGlue: One command to local AI coding bliss!** 🚀

*Get the power of Qwen3's direct, practical responses running locally on your machine in minutes.*
