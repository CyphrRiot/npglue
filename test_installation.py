#!/usr/bin/env python3
"""
Simple test to verify Qwen3-8B OpenVINO installation
"""

import time
import psutil
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

def test_model():
    print("🧪 Testing DeepSeek-R1 FP16 OpenVINO Installation")
    print("=" * 50)
    
    model_path = "models/deepseek-r1-fp16-ov/DeepSeek-R1-0528-Qwen3-8B-fp16-ov"
    
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
    prompt = "def hello_world():"
    print(f"Prompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    start_gen = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,  # Deterministic for testing
        pad_token_id=tokenizer.eos_token_id
    )
    gen_time = time.time() - start_gen
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = response[len(prompt):].strip()
    
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    speed = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"Generated: '{generated[:80]}{'...' if len(generated) > 80 else ''}'")
    print(f"Speed: {speed:.1f} tokens/sec ({tokens_generated} tokens in {gen_time:.2f}s)")
    
    # Performance assessment
    if speed >= 10:
        status = "✅ EXCELLENT"
    elif speed >= 6:
        status = "✅ GOOD"
    elif speed >= 3:
        status = "✅ OK"
    else:
        status = "⚠️ SLOW"
    
    print(f"Status: {status}")
    
    return speed > 3

if __name__ == "__main__":
    try:
        success = test_model()
        print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Installation test complete")
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
