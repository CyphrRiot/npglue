# NPGlue - Local AI Development Assistant

**NPGlue** provides a complete setup for running **DeepSeek-R1** locally using OpenVINO for AI-assisted coding and development.

## 🚀 **Quick Start**

```bash
git clone https://github.com/CyphrRiot/npglue.git
cd npglue
./install
```

That's it! The installer does everything and gives you clear instructions for Goose and Zed.

## ✅ **What You Get**

- **DeepSeek-R1**: Superior reasoning AI model
- **INT4-AWQ Optimized**: Advanced quantization preserving 95%+ quality
- **20-30+ tokens/sec**: Fast local inference with reduced RAM usage
- **Memory Efficient**: ~6-8GB RAM usage (vs 10-12GB FP16)
- **Goose Compatible**: Drop-in replacement for OpenAI
- **Zed Ready**: Perfect for local AI coding

## 🔧 **Requirements**

- **OS**: Linux (Arch/CachyOS recommended)
- **Memory**: 8GB+ RAM (12GB+ recommended)
- **Storage**: 15GB free space
- **CPU**: Intel preferred (Core Ultra optimal)

## 📊 **Performance**

| Speed | Memory | Model Size | Latency |
|-------|--------|------------|---------|
| 20-30+ tok/s | ~6-8GB | ~5.6GB INT4-AWQ | <1s first token |

## 🦆 **Works With**

- **Goose**: AI development assistant
- **Zed**: Modern code editor
- **Cursor**: AI-powered IDE
- **Any OpenAI-compatible tool**

## 🛠️ **After Installation**

The installer gives you exact instructions for:
1. Starting the server
2. Configuring Goose
3. Setting up Zed
4. Testing everything

## 📁 **What Gets Installed**

```
npglue/
├── install                 # Beautiful installer script
├── start_server.sh         # Start the AI server
├── server_production.py    # FastAPI server
├── test_installation.py    # Verify everything works
├── openvino-env/          # Python environment
├── models/                # DeepSeek-R1 INT4-AWQ model
```

## 🎯 **Why NPGlue?**

- **One Command**: `./install` does everything
- **Memory Safe**: Won't crash during setup
- **Professional**: Uses expert-optimized models
- **Beautiful Output**: Clear, colorful installer
- **Complete Instructions**: Tells you exactly what to do next

---

**NPGlue: One command to local AI coding bliss!** 🚀
