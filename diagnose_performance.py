#!/usr/bin/env python3
"""
Performance diagnostics for NPGlue
Helps identify why token generation might be slow
"""

import psutil
import os
import time
import subprocess

def check_cpu_governor():
    """Check CPU scaling governor"""
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
            governor = f.read().strip()
        print(f"🔧 CPU Governor: {governor}")
        
        if governor != 'performance':
            print("⚠️  WARNING: CPU not in performance mode!")
            print("   Run: sudo ./boost_cpu.sh")
        else:
            print("✅ CPU in performance mode")
            
    except FileNotFoundError:
        print("⚠️  Can't read CPU governor (may be VM or different system)")

def check_cpu_frequency():
    """Check current CPU frequencies"""
    try:
        freqs = psutil.cpu_freq(percpu=True)
        if freqs:
            avg_freq = sum(f.current for f in freqs) / len(freqs)
            max_freq = max(f.max for f in freqs)
            print(f"⚡ CPU Frequency: {avg_freq:.0f} MHz (max: {max_freq:.0f} MHz)")
            
            if avg_freq < max_freq * 0.8:
                print("⚠️  WARNING: CPU running below 80% max frequency!")
        else:
            print("⚠️  Can't read CPU frequency")
    except:
        print("⚠️  Error reading CPU frequency")

def check_memory_usage():
    """Check memory and swap usage"""
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"💾 Memory: {mem.used/1024**3:.1f}GB used / {mem.total/1024**3:.1f}GB total ({mem.percent:.1f}%)")
    print(f"💿 Swap: {swap.used/1024**3:.1f}GB used / {swap.total/1024**3:.1f}GB total ({swap.percent:.1f}%)")
    
    if mem.percent > 90:
        print("❌ WARNING: Very high memory usage!")
    elif mem.percent > 80:
        print("⚠️  WARNING: High memory usage")
    else:
        print("✅ Memory usage looks good")
        
    if swap.percent > 10:
        print("❌ WARNING: System is swapping to disk!")
        print("   This will severely slow down AI inference")

def check_cpu_load():
    """Check CPU load and processes"""
    load1, load5, load15 = os.getloadavg()
    cpu_count = psutil.cpu_count()
    
    print(f"🏋️  CPU Load: {load1:.2f} (1min) {load5:.2f} (5min) {load15:.2f} (15min)")
    print(f"🔀 CPU Cores: {cpu_count}")
    
    if load1 > cpu_count * 0.8:
        print("⚠️  WARNING: High CPU load!")
        
    # Show top CPU consuming processes
    processes = [(p.info['pid'], p.info['name'], p.info['cpu_percent']) 
                for p in psutil.process_iter(['pid', 'name', 'cpu_percent'])]
    processes.sort(key=lambda x: x[2], reverse=True)
    
    print("\n🔥 Top CPU processes:")
    for pid, name, cpu in processes[:5]:
        if cpu > 1.0:
            print(f"   {name} (PID {pid}): {cpu:.1f}%")

def check_model_config():
    """Check which model is configured"""
    try:
        with open('.model_config', 'r') as f:
            config = f.read().strip()
            model_path = config.split('=')[1]
        
        print(f"🤖 Model Config: {model_path}")
        
        if "8b-int8" in model_path.lower():
            print("📊 Using 8B model - needs ~6-8GB RAM")
        elif "0.6b" in model_path.lower():
            print("📊 Using 0.6B model - needs ~1-2GB RAM")
        else:
            print("❓ Unknown model variant")
            
    except FileNotFoundError:
        print("❌ No .model_config found - model path unknown")

def check_openvino():
    """Check OpenVINO setup"""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"🧠 OpenVINO devices: {devices}")
        
        if 'CPU' not in devices:
            print("❌ WARNING: CPU not available in OpenVINO!")
        else:
            print("✅ OpenVINO CPU support available")
            
    except ImportError:
        print("❌ OpenVINO not installed or not in Python path")

def run_quick_benchmark():
    """Run a quick computation benchmark"""
    print("\n🏃 Running quick CPU benchmark...")
    
    start_time = time.time()
    # Simple CPU-intensive task
    result = sum(i**2 for i in range(1000000))
    end_time = time.time()
    
    benchmark_time = end_time - start_time
    print(f"⏱️  CPU Benchmark: {benchmark_time:.3f} seconds")
    
    if benchmark_time > 0.5:
        print("⚠️  CPU seems slower than expected")
    else:
        print("✅ CPU performance looks normal")

def main():
    print("🔍 NPGlue Performance Diagnostics")
    print("=" * 40)
    
    check_cpu_governor()
    print()
    
    check_cpu_frequency() 
    print()
    
    check_memory_usage()
    print()
    
    check_cpu_load()
    print()
    
    check_model_config()
    print()
    
    check_openvino()
    
    run_quick_benchmark()
    
    print("\n💡 Performance Tips:")
    print("1. Make sure CPU is in performance mode: sudo ./boost_cpu.sh")
    print("2. Close other heavy applications")
    print("3. Check if you need the 0.6B model instead of 8B")
    print("4. Ensure system isn't swapping to disk")
    print("5. Restart the server after system changes")

if __name__ == "__main__":
    main()
