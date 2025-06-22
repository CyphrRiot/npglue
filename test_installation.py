#!/usr/bin/env python3
"""
Simple test to verify Qwen3 OpenVINO installation and performance display
"""

import time
import psutil
import requests
import json
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def test_model_direct():
    print("🧪 Testing Qwen3 OpenVINO Installation (Direct)")
    print("=" * 50)
    
    # Read model path from config
    try:
        with open('.model_config', 'r') as f:
            config = f.read().strip()
            model_path = config.split('=')[1]
        print(f"📂 Using model: {model_path}")
    except FileNotFoundError:
        model_path = "models/qwen3-8b-int8"
        print(f"📂 Using default model: {model_path}")
    
    print("📥 Loading model and tokenizer...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = OVModelForCausalLM.from_pretrained(model_path, device="CPU")
    
    load_time = time.time() - start_time
    memory_mb = psutil.Process().memory_info().rss / (1024**2)
    
    print(f"✅ Loaded in {load_time:.1f}s, using {memory_mb/1024:.1f}GB")
    
    # Quick test
    print("\n🎯 Testing generation...")
    prompt = "What is 2 + 2?"
    print(f"Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_gen = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_time = time.time() - start_gen
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    speed = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"Generated: '{generated[:80]}{'...' if len(generated) > 80 else ''}'")
    print(f"Speed: {speed:.1f} tokens/sec ({tokens_generated} tokens in {gen_time:.2f}s)")
    
    return speed > 3

def test_server_api():
    print("\n🌐 Testing Server API and Performance Display")
    print("=" * 50)
    
    server_url = "http://localhost:11434"
    
    # Check if server is running
    try:
        health = requests.get(f"{server_url}/health", timeout=5)
        print(f"✅ Server health: {health.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: ./start_server.sh")
        return False
    except Exception as e:
        print(f"⚠️ Health check failed: {e}")
        return False
    
    # Test OpenAI-compatible endpoint
    print("\n🔍 Testing OpenAI endpoint (/v1/chat/completions)...")
    test_request = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=test_request,
            timeout=30
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ Response received in {request_time:.2f}s")
            print(f"Content: {content[:150]}{'...' if len(content) > 150 else ''}")
            
            # Check if performance footer is present
            if "Completed in" in content and "tokens/sec" in content:
                print("✅ Performance display working correctly!")
                return True
            else:
                print("⚠️ Performance display not found in response")
                return False
        else:
            print(f"❌ API call failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_ollama_api():
    print("\n🦙 Testing Ollama-compatible endpoint (/api/chat)...")
    server_url = "http://localhost:11434"
    
    test_request = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Hello, can you help me?"}],
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/api/chat",
            json=test_request,
            timeout=30
        )
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            content = result["message"]["content"]
            print(f"✅ Ollama response received in {request_time:.2f}s")
            print(f"Content: {content[:150]}{'...' if len(content) > 150 else ''}")
            
            # Check if performance footer is present
            if "Completed in" in content and "tokens/sec" in content:
                print("✅ Performance display working correctly!")
                return True
            else:
                print("⚠️ Performance display not found in response")
                return False
        else:
            print(f"❌ Ollama API call failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ollama API test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        print("🚀 NPGlue Installation & Performance Test")
        print("=" * 50)
        
        # Test direct model loading
        direct_success = test_model_direct()
        
        # Test server APIs
        api_success = test_server_api()
        ollama_success = test_ollama_api()
        
        overall_success = direct_success and (api_success or ollama_success)
        
        print(f"\n📊 Test Results:")
        print(f"Direct model: {'✅ PASS' if direct_success else '❌ FAIL'}")
        print(f"OpenAI API: {'✅ PASS' if api_success else '❌ FAIL'}")
        print(f"Ollama API: {'✅ PASS' if ollama_success else '❌ FAIL'}")
        print(f"\n{'✅ SUCCESS' if overall_success else '❌ FAILED'}: Installation test complete")
        
        if overall_success:
            print("\n💡 Tips:")
            print("- Your local AI is working with performance display!")
            print("- Use any OpenAI-compatible client with http://localhost:11434")
            print("- Or use as Ollama drop-in replacement")
            print("- All responses now show completion time and token rate")
        
        exit(0 if overall_success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
