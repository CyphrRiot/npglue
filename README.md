# NPGlue - Local AI Development Assistant

**NPGlue** provides a complete setup for running **DeepSeek-R1** locally using OpenVINO for AI-assisted coding and development.

## ✅ **What This Provides**

- **Advanced AI Model**: DeepSeek-R1 (superior reasoning capabilities)
- **FP16 Precision**: Professional OpenVINO conversion for optimal speed
- **Fast Inference**: 15-25+ tokens/sec on Intel Core Ultra CPUs
- **Memory Efficient**: ~10-12GB RAM usage
- **OpenAI-Compatible API**: Works with Goose, Cursor, and other tools
- **Production Ready**: FastAPI server with proper error handling

## 🚀 **Quick Start**

```bash
git clone <your-repo-url> npglue
cd npglue
chmod +x install.sh
./install.sh
```

After installation:
```bash
./start_server.sh  # Start the AI server
```

## 🔧 **System Requirements**

- **OS**: Linux (tested on Arch/CachyOS)
- **Memory**: 12GB+ RAM recommended
- **Storage**: 15GB free disk space  
- **CPU**: Intel processors recommended (Core Ultra series optimal)
- **Optional**: Intel NPU for potential acceleration

## 📊 **Performance Expectations**

| Metric | Value |
|--------|-------|
| **Speed** | 15-25+ tokens/sec (FP16 optimized) |
| **Memory** | 10-12GB peak usage |
| **Model Size** | ~8GB (FP16 precision) |
| **First Run** | ~8-10 tokens/sec (cold start) |
| **Warmed Up** | ~20+ tokens/sec |
| **Latency** | <1 second first token |

## 🦆 **Goose Integration**

NPGlue works seamlessly with Goose AI assistant. Copy the example config:

```bash
cp goose_config_example.yaml ~/.config/goose/config.yaml
```

Or manually add to your Goose config:

```yaml
GOOSE_MODEL: deepseek-r1-fp16-openvino
GOOSE_PROVIDER: openai
GOOSE_API_BASE: http://localhost:8000/v1
GOOSE_API_KEY: local-key
```

Start NPGlue server, then use Goose normally - it will use your local model!

## 🛠️ **Installation Details**

The installer automatically:

1. **System Dependencies**: Python, cmake, git, Intel drivers
2. **Virtual Environment**: Clean Python environment with OpenVINO
3. **Model Download**: Pre-converted DeepSeek-R1 FP16 OpenVINO model
4. **Performance Optimization**: CPU performance tuning
5. **Testing**: Verification that everything works

## 🖥️ **Usage**

### Start the Server
```bash
./start_server.sh
```

The server runs on `http://localhost:8000` with:
- OpenAI-compatible API at `/v1/chat/completions`
- Health check at `/health`
- API documentation at `/docs`

### Test Installation
```bash
python test_installation.py
```

### API Example
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-openvino",
    "messages": [
      {"role": "user", "content": "Write a Python function to calculate fibonacci"}
    ],
    "max_tokens": 150
  }'
```

## 🔍 **Troubleshooting**

### Performance Issues
- First run is always slower (warmup effect)
- Ensure CPU performance mode: `sudo ./boost_cpu.sh`
- Check available memory: `free -h`

### Memory Errors
- Model requires 10-12GB RAM during inference
- Close other applications if needed
- Server has built-in memory monitoring

### Model Loading Issues
```bash
# Verify OpenVINO installation
source openvino-env/bin/activate
python -c "import openvino; print(openvino.Core().available_devices)"
```

## 📁 **Project Structure**

```
npglue/
├── install.sh              # Main installation script
├── start_server.sh          # Start the AI server
├── server_production.py     # FastAPI server with OpenAI API
├── test_installation.py     # Installation verification
├── boost_cpu.sh            # CPU performance optimization
├── goose_config_example.yaml # Goose configuration template
├── README.md               # This file
├── openvino-env/           # Python virtual environment (created)
└── models/                 # Downloaded models (created)
    └── deepseek-r1-fp16-ov/
```

## 🔧 **For Developers**

### Extending NPGlue

The server provides OpenAI-compatible endpoints, making it compatible with:
- **Cursor IDE**: Point to `http://localhost:8000`
- **Continue.dev**: Configure as OpenAI provider
- **Any OpenAI-compatible client**

### Configuration

Edit `server_production.py` to modify:
- Memory limits and monitoring
- Generation parameters
- API endpoints and responses
- Device selection (CPU/GPU/NPU)

## 🎯 **Why NPGlue?**

- **Local Control**: No data sent to external APIs
- **Fast Performance**: Optimized for Intel hardware
- **Professional Quality**: Uses expert-converted OpenVINO models
- **Easy Integration**: Drop-in replacement for OpenAI API
- **Production Ready**: Proper error handling and monitoring

## 📈 **Status**

- ✅ **CPU Inference**: Fully working, 15-25+ tok/s
- ✅ **Memory Management**: Stable, no crashes
- ✅ **OpenAI API**: Compatible with standard tools
- ✅ **Goose Integration**: Plug-and-play configuration
- ⚠️ **NPU Support**: Experimental, fallback to CPU
- ✅ **Production Ready**: Error handling, monitoring

---

**NPGlue: Seamlessly connect your development environment to local AI!** 🚀
