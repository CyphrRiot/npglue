#!/usr/bin/env python3
"""
Debug empty response issue
Test various endpoint combinations to identify the problem
"""

import requests
import json
import time

def test_endpoint(endpoint, data, description):
    print(f"\n🔍 Testing {description}")
    print(f"   Endpoint: {endpoint}")
    print(f"   Data: {json.dumps(data, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(f"http://localhost:11434{endpoint}", 
                               json=data, 
                               timeout=30)
        request_time = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time: {request_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract content based on endpoint
            if "/v1/chat/completions" in endpoint:
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif "/api/chat" in endpoint:
                content = result.get("message", {}).get("content", "")
            elif "/api/generate" in endpoint:
                content = result.get("response", "")
            else:
                content = str(result)
            
            print(f"   Content length: {len(content)}")
            if content:
                print(f"   Content preview: {content[:200]}...")
                if "*Completed in" in content:
                    print("   ✅ Performance footer present")
                else:
                    print("   ❌ Performance footer missing")
            else:
                print("   ❌ EMPTY CONTENT!")
                print(f"   Full response: {json.dumps(result, indent=2)}")
                
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

def main():
    print("🐛 NPGlue Empty Response Debug")
    print("=" * 40)
    
    # Test different endpoints with same simple prompt
    simple_prompt = "What is 2+2?"
    
    # Test OpenAI chat completions
    test_endpoint("/v1/chat/completions", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": simple_prompt}],
        "max_tokens": 50
    }, "OpenAI Chat Completions (50 tokens)")
    
    # Test with more tokens
    test_endpoint("/v1/chat/completions", {
        "model": "qwen3", 
        "messages": [{"role": "user", "content": simple_prompt}],
        "max_tokens": 200
    }, "OpenAI Chat Completions (200 tokens)")
    
    # Test Ollama chat
    test_endpoint("/api/chat", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": simple_prompt}]
    }, "Ollama Chat")
    
    # Test Ollama generate
    test_endpoint("/api/generate", {
        "model": "qwen3",
        "prompt": simple_prompt
    }, "Ollama Generate")
    
    # Test with different prompts
    print(f"\n🔍 Testing different prompt types...")
    
    test_endpoint("/api/chat", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Hello"}]
    }, "Simple greeting")
    
    test_endpoint("/api/chat", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "Write a Python function to add two numbers"}]
    }, "Code request")
    
    # Test edge cases
    test_endpoint("/api/chat", {
        "model": "qwen3",
        "messages": [{"role": "user", "content": ""}]
    }, "Empty prompt")
    
    print(f"\n📊 Debug Summary:")
    print("If some tests show empty content while others work,")
    print("the issue is likely in specific request handling logic.")

if __name__ == "__main__":
    main()
