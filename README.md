# NPGlue - Intel NPU Glue for AI

**NPGlue** provides a complete setup for running **DeepSeek-R1** locally using OpenVINO for AI-assisted coding and development.

## 🚀 **Quick Start**

```bash
git clone https://github.com/CyphrRiot/npglue.git
cd npglue
./install
```

That's it! The beautiful installer does everything and gives you clear, safe instructions for Goose and Zed.

## ✅ **What You Get**

- **DeepSeek-R1**: Superior reasoning AI model with advanced thinking capabilities
- **INT4-AWQ Optimized**: Advanced quantization preserving 95%+ quality at half the size
- **20-30+ tokens/sec**: Fast local inference with reduced RAM usage
- **Memory Efficient**: ~6-8GB RAM usage (vs 10-12GB FP16)
- **Zed Compatible**: Works as Ollama provider (no API key hassles!)
- **Dual API Support**: Both OpenAI and Ollama compatible endpoints
- **Goose Ready**: Drop-in replacement for OpenAI API

## 🔧 **Requirements**

- **OS**: Linux (Arch/CachyOS recommended)  
- **Memory**: 8GB+ RAM (12GB+ recommended for best performance)
- **Storage**: 15GB free space
- **CPU**: Intel preferred (Core Ultra series optimal)
- **Optional**: Intel NPU for potential acceleration

## 📊 **Performance Expectations**

| Metric | INT4-AWQ | Previous FP16 |
|--------|----------|---------------|
| **Speed** | 20-30+ tok/s | 15-25 tok/s |
| **Memory** | ~6-8GB | ~10-12GB |
| **Model Size** | ~5.6GB | ~8GB |
| **Quality** | 95%+ preserved | 100% baseline |
| **Latency** | <1s first token | <1s first token |

## 🛠️ **What the Installer Does**

### **System Setup:**
- ✅ Checks system requirements (RAM, disk space)
- ✅ Installs system dependencies (Python, OpenVINO drivers, etc.)
- ✅ Creates clean Python virtual environment
- ✅ Installs AI packages (OpenVINO 2024.x, transformers, etc.)

### **Model Setup:**
- ✅ Downloads optimized DeepSeek-R1 INT4-AWQ model (~5.6GB)
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
model: deepseek-r1-openvino  
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
          "name": "deepseek-r1-openvino",
          "display_name": "DeepSeek-R1 Local",
          "max_tokens": 32768,
          "supports_tools": true
        }
      ]
    }
  },
  "agent": {
    "default_model": {
      "provider": "ollama",
      "model": "deepseek-r1-openvino"
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
├── openvino-env/              # Python environment (created)
└── models/                    # INT4-AWQ model (downloaded)
    └── deepseek-r1-int4-awq/
```

## 🎯 **Why Choose NPGlue?**

- **One Command Setup**: `./install` does everything beautifully
- **Memory Safe**: Won't crash during installation or use
- **Configuration Safe**: Won't overwrite your existing tool settings  
- **Professional Quality**: Uses expert-optimized INT4-AWQ models
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
source openvino-env/bin/activate

# Check available devices
python -c "import openvino; print(openvino.Core().available_devices)"
```

## 🚀 **Recent Improvements**

- ✅ **Dual API Support**: Both OpenAI AND Ollama compatible endpoints
- ✅ **Zed Integration Fixed**: Works as Ollama provider (no API key issues!)  
- ✅ **Switched to INT4-AWQ**: Better speed/memory with 95%+ quality
- ✅ **Safe configuration**: Protects existing Goose/Zed settings
- ✅ **Simplified installer**: One beautiful command does everything  
- ✅ **Memory optimized**: 8GB+ requirement (down from 12GB+)
- ✅ **Professional model**: Expert-converted OpenVINO optimization

---

**NPGlue: One command to local AI coding bliss!** 🚀

*Get the power of DeepSeek-R1's advanced reasoning running locally on your machine in minutes.*
