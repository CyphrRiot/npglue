#!/usr/bin/env python3
"""
Production-ready Qwen3-8B OpenVINO Server
Memory-safe with proper error handling
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import openvino as ov
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer
import uvicorn
import threading
import psutil
import os
import gc
import time
from typing import Optional

# Global variables
model = None
tokenizer = None
device_used = None
model_lock = threading.Lock()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    stream: bool = False

class GenerateResponse(BaseModel):
    response: str
    device: str
    memory_gb: float
    generation_time: float
    words_per_sec: float

app = FastAPI(
    title="NPGlue - Qwen3-8B OpenVINO Server",
    description="Local AI development assistant with Qwen3-8B for direct answers",
    version="1.0.0"
)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def check_memory_availability():
    """Check if we have enough memory for operation"""
    available_gb = psutil.virtual_memory().available / (1024**3)
    current_usage_gb = get_memory_usage()
    
    # Need at least 0.8GB free after current usage for generation
    return available_gb > 0.8, available_gb, current_usage_gb

def load_model_safe():
    """Load model with memory monitoring and error handling"""
    global model, tokenizer, device_used
    
    if model is not None:
        return model, tokenizer, device_used
    
    with model_lock:
        if model is not None:  # Double-check pattern
            return model, tokenizer, device_used
        
        # Read model path from config file
        try:
            with open('.model_config', 'r') as f:
                config = f.read().strip()
                model_path = config.split('=')[1]
            print(f"🤖 Loading {model_path.split('/')[-1]} OpenVINO model...")
        except FileNotFoundError:
            # Fallback to default
            model_path = "models/qwen3-8b-int8"
            print("🤖 Loading Qwen3-8B INT8 OpenVINO model...")
        except Exception:
            model_path = "models/qwen3-8b-int8"
            print("🤖 Loading Qwen3-8B INT8 OpenVINO model...")
        
        # Check memory before loading
        has_memory, available_gb, current_gb = check_memory_availability()
        if not has_memory:
            raise HTTPException(
                status_code=503, 
                detail=f"Insufficient memory. Available: {available_gb:.1f}GB, Current: {current_gb:.1f}GB"
            )
        
        try:
            # Load tokenizer first
            print("  Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            tokenizer_memory = get_memory_usage()
            print(f"  Tokenizer loaded ({tokenizer_memory:.1f}GB)")
            
            # Load model on CPU (most stable)
            print("  Loading model on CPU...")
            model = OVModelForCausalLM.from_pretrained(
                model_path,
                device="CPU",
                trust_remote_code=True
            )
            device_used = "CPU"
            
            model_memory = get_memory_usage()
            print(f"  Model loaded on {device_used} ({model_memory:.1f}GB total)")
            
            # Test generation to ensure it works
            print("  Testing model...")
            test_inputs = tokenizer("Hello", return_tensors="pt")
            test_outputs = model.generate(**test_inputs, max_new_tokens=5, do_sample=False)
            test_response = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
            print(f"  Test successful: '{test_response[:50]}...'")
            
            return model, tokenizer, device_used
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Cleanup on failure
            if 'model' in locals() and model is not None:
                del model
                model = None
            if 'tokenizer' in locals() and tokenizer is not None:
                del tokenizer
                tokenizer = None
            gc.collect()
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check with memory status"""
    try:
        has_memory, available_gb, current_gb = check_memory_availability()
        
        # Check if model is loaded
        model_status = "loaded" if model is not None else "not_loaded"
        
        return {
            "status": "healthy" if has_memory else "low_memory",
            "model_status": model_status,
            "device": device_used,
            "memory": {
                "current_gb": current_gb,
                "available_gb": available_gb,
                "sufficient": has_memory
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """OpenAI-compatible chat completions endpoint for Goose and Zed - NO AUTH REQUIRED"""
    try:
        print(f"🔍 DEBUG: Request received: {list(request.keys())}")
        
        # Extract message content from OpenAI format
        messages = request.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last user message
        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Extract parameters
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.7)
        
        # Load model if needed
        model, tokenizer, device = load_model_safe()
        
        # Check memory before generation
        has_memory, available_gb, current_gb = check_memory_availability()
        if not has_memory:
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient memory for generation. Available: {available_gb:.1f}GB"
            )
        
        # Limit max tokens to prevent memory issues
        max_tokens = min(max_tokens, 200)
        
        # Tokenize input
        inputs = tokenizer(
            user_message,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024
        )
        input_length = inputs.input_ids.shape[1]  # Get input token length
        
        # Generate
        generation_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=min(max(temperature, 0.1), 2.0),
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            output_attentions=False,
            output_hidden_states=False
        )
        generation_time = time.time() - generation_start
        
        # Properly extract only the generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Cleanup
        del outputs, inputs
        gc.collect()
        
        # Return OpenAI-compatible format
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen3", 
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]) if 'inputs' in locals() else 0,
                "completion_tokens": len(tokenizer.encode(generated_text)),
                "total_tokens": len(inputs.input_ids[0]) + len(tokenizer.encode(generated_text)) if 'inputs' in locals() else len(tokenizer.encode(generated_text))
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text with memory monitoring"""
    start_time = time.time()
    
    try:
        # Load model if needed
        model, tokenizer, device = load_model_safe()
        
        # Check memory before generation
        has_memory, available_gb, current_gb = check_memory_availability()
        if not has_memory:
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient memory for generation. Available: {available_gb:.1f}GB"
            )
        
        # Limit max tokens to prevent memory issues
        max_tokens = min(request.max_tokens, 200)
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024  # Limit context length
        )
        input_length = inputs.input_ids.shape[1]  # Get input token length
        
        # Generate with safe parameters
        generation_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=min(max(request.temperature, 0.1), 2.0),  # Clamp temperature
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Performance settings
            use_cache=True,
            num_beams=1,  # Greedy decoding
            early_stopping=True,
            # Memory optimization
            output_attentions=False,
            output_hidden_states=False
        )
        generation_time = time.time() - generation_start
        
        # Properly extract only the generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Calculate metrics
        word_count = len(generated_text.split())
        words_per_sec = word_count / generation_time if generation_time > 0 else 0
        total_time = time.time() - start_time
        final_memory = get_memory_usage()
        
        # Cleanup
        del outputs
        del inputs
        gc.collect()
        
        return GenerateResponse(
            response=generated_text,
            device=device,
            memory_gb=final_memory,
            generation_time=generation_time,
            words_per_sec=words_per_sec
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        # Force memory cleanup on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models and system info"""
    try:
        core = ov.Core()
        has_memory, available_gb, current_gb = check_memory_availability()
        
        return {
            "models": ["qwen3"],
            "devices": core.available_devices,
            "current_device": device_used,
            "model_loaded": model is not None,
            "memory": {
                "current_gb": current_gb,
                "available_gb": available_gb,
                "sufficient": has_memory
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/v1/models") 
async def list_models_openai():
    """OpenAI-compatible models endpoint for Zed - NO AUTH REQUIRED"""
    try:
        print(f"🔍 DEBUG: /v1/models called")
        
        # Return OpenAI-compatible models list
        return {
            "object": "list",
            "data": [
                {
                    "id": "qwen3",
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": "local",
                    "permission": [],
                    "root": "qwen3",
                    "parent": None
                }
            ]
        }
    except Exception as e:
        return {"error": str(e)}

# Add Ollama-compatible endpoints
@app.get("/api/tags")
async def ollama_list_models():
    """Ollama-compatible models list endpoint"""
    try:
        print(f"🔍 DEBUG: /api/tags called (Ollama models list)")
        return {
            "models": [
                {
                    "name": "qwen3",
                    "size": 6000000000,  # ~6GB
                    "digest": "local",
                    "model": "qwen3",
                    "modified_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "details": {
                        "format": "openvino",
                        "family": "qwen",
                        "families": ["qwen"],
                        "parameter_size": "8B",
                        "quantization_level": "INT8"
                    }
                }
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/generate")
async def ollama_generate(request: dict):
    """Ollama-compatible generate endpoint"""
    try:
        print(f"🔍 DEBUG: /api/generate called with: {list(request.keys())}")
        
        # Extract prompt from Ollama format
        prompt = request.get("prompt", "")
        model_name = request.get("model", "")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="No prompt provided")
        
        print(f"🔍 DEBUG: Model: {model_name}, Prompt: {prompt[:50]}...")
        
        # Load model if needed
        model, tokenizer, device = load_model_safe()
        
        # Check memory before generation
        has_memory, available_gb, current_gb = check_memory_availability()
        if not has_memory:
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient memory for generation. Available: {available_gb:.1f}GB"
            )
        
        # Generate with strict memory limits
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_length = inputs.input_ids.shape[1]  # Get input token length
        
        # Check memory before generation (be more reasonable)
        has_memory, available_gb, current_gb = check_memory_availability()
        if available_gb < 0.8:  # Only need 800MB free
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient memory for generation. Need 0.8GB+, have {available_gb:.1f}GB"
            )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # Reasonable length
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1
        )
        
        # Properly extract only the generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Cleanup
        del outputs, inputs
        gc.collect()
        
        # Return Ollama-compatible format
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "response": generated_text,
            "done": True,
            "context": [],
            "total_duration": 1000000000,  # 1 second in nanoseconds
            "load_duration": 0,
            "prompt_eval_count": len(tokenizer.encode(prompt)),
            "prompt_eval_duration": 500000000,
            "eval_count": len(tokenizer.encode(generated_text)),
            "eval_duration": 500000000
        }
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def ollama_chat(request: dict):
    """Ollama-compatible chat endpoint for Zed"""
    try:
        print(f"🔍 DEBUG: /api/chat called with: {list(request.keys())}")
        
        # Extract from Ollama chat format
        messages = request.get("messages", [])
        model_name = request.get("model", "")
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        print(f"🔍 DEBUG: Chat - Model: {model_name}, User message: {user_message[:50]}...")
        
        # Load model if needed
        model, tokenizer, device = load_model_safe()
        
        # Simple direct prompting - Qwen3 should handle this well
        # Generate response with strict memory limits
        inputs = tokenizer(user_message, return_tensors="pt", truncation=True, max_length=512)
        input_length = inputs.input_ids.shape[1]  # Get input token length
        
        # Check memory before generation (be more reasonable)  
        has_memory, available_gb, current_gb = check_memory_availability()
        if available_gb < 0.8:  # Only need 800MB free for generation
            raise HTTPException(
                status_code=503,
                detail=f"Insufficient memory for generation. Need 0.8GB+, have {available_gb:.1f}GB"
            )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,  # Reasonable length for answers
            do_sample=True,
            temperature=0.7,    # Normal temperature for natural responses
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
        )
        
        # Properly extract only the generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Cleanup
        del outputs, inputs
        gc.collect()
        
        # Return Ollama chat format
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "done": True
        }
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload")
async def unload_model():
    global model, tokenizer, device_used
    
    with model_lock:
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        device_used = None
        gc.collect()
    
    return {"status": "model_unloaded", "memory_gb": get_memory_usage()}

if __name__ == "__main__":
    import sys
    
    print("🚀 Starting NPGlue - Qwen3-8B OpenVINO Server")
    print("=" * 50)
    
    # System info
    print(f"💾 Memory: {psutil.virtual_memory().total // (1024**3)}GB total")
    print(f"🔧 CPU: {psutil.cpu_count()} cores")
    
    # OpenVINO info
    try:
        core = ov.Core()
        print(f"🤖 OpenVINO devices: {core.available_devices}")
    except Exception as e:
        print(f"⚠️  OpenVINO error: {e}")
    
    # Check initial memory
    has_memory, available_gb, current_gb = check_memory_availability()
    print(f"📊 Memory status: {current_gb:.1f}GB used, {available_gb:.1f}GB available")
    
    if not has_memory:
        print("❌ Insufficient memory to start server")
        sys.exit(1)
    
    print("\n🔄 Starting server on http://127.0.0.1:11434")
    print("📖 Visit http://127.0.0.1:11434/docs for API documentation")
    
    # Start the server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=11434,
        log_level="info"
    )
