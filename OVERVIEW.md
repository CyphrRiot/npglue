# NPGlue Project Overview

**NPGlue** (Neural Processing Glue) - Clean, production-ready local AI development assistant.

## 🎯 **What NPGlue Does**

NPGlue seamlessly connects your development environment to a powerful local AI model (DeepSeek-R1) running through OpenVINO optimization. No cloud dependencies, no data leaks, just fast local AI assistance.

## 📁 **Project Files**

### Core Files
- `install.sh` - Complete automated installation
- `start_server.sh` - Simple server startup  
- `server_production.py` - FastAPI server with OpenAI-compatible API
- `test_installation.py` - Installation verification

### Configuration & Optimization  
- `goose_config_example.yaml` - Ready-to-use Goose configuration
- `boost_cpu.sh` - CPU performance optimization
- `.gitignore` - Excludes generated files (models, venv)

### Documentation
- `README.md` - Complete setup and usage guide
- `OVERVIEW.md` - This file

## 🚀 **Quick Start for New Users**

1. **Clone the repo**: `git clone <url> npglue`
2. **Run installer**: `cd npglue && ./install.sh` 
3. **Start server**: `./start_server.sh`
4. **Configure Goose**: Copy `goose_config_example.yaml` to `~/.config/goose/config.yaml`

## 🎯 **Design Principles**

- **Minimal Dependencies**: Only essential files, no bloat
- **Self-Contained**: All dependencies installed in virtual environment
- **Production Ready**: Proper error handling and memory management
- **Tool Agnostic**: OpenAI-compatible API works with any client
- **Clean Structure**: Easy to understand and modify

## 📊 **Performance Targets**

- **Speed**: 15-25+ tokens/sec (FP16 optimized model)
- **Memory**: 10-12GB peak (stable, no crashes)
- **Latency**: <1 second first token
- **Reliability**: Production-grade error handling

## 🔧 **Technical Stack**

- **Model**: DeepSeek-R1-0528-Qwen3-8B (FP16 OpenVINO)
- **Runtime**: OpenVINO 2025.x on CPU/NPU
- **Server**: FastAPI with OpenAI-compatible endpoints
- **Optimization**: Intel CPU performance tuning

---

**NPGlue: Simple, Fast, Local AI Development** 🚀
