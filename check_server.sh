#!/bin/bash
# Quick NPGlue server status check

echo "🔍 NPGlue Server Status Check"
echo "============================="

# Check if server process is running
SERVER_PID=$(pgrep -f "server_production.py")
if [ -n "$SERVER_PID" ]; then
    echo "✅ Server running (PID: $SERVER_PID)"
    
    # Check memory usage
    MEMORY=$(ps -p $SERVER_PID -o rss= | awk '{print $1/1024 " MB"}')
    echo "💾 Memory usage: $MEMORY"
    
    # Test health endpoint
    echo "🏥 Testing health endpoint..."
    HEALTH=$(curl -s http://localhost:11434/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "✅ Health endpoint responding"
        echo "   Response: $HEALTH"
    else
        echo "❌ Health endpoint not responding"
    fi
    
    # Test model list
    echo "🤖 Testing model list..."
    MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "✅ Models endpoint responding"
    else
        echo "❌ Models endpoint not responding"
    fi
else
    echo "❌ Server not running"
    echo "💡 Start with: ./start_server.sh"
fi

echo
echo "📊 Current configuration:"
if [ -f ".model_config" ]; then
    cat .model_config
else
    echo "   No model config found"
fi

echo
echo "📂 Available models:"
if [ -d "models" ]; then
    ls -la models/
else
    echo "   No models directory found"
fi
