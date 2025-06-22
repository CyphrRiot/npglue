#!/usr/bin/env python3
"""
NPGlue Simple Setup Script
=========================
This script does everything:
1. Checks for the model
2. Installs dependencies 
3. Runs a test
4. Tells you what to do next
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(text):
    print(f"\n🚀 {text}")
    print("=" * (len(text) + 3))

def print_step(text):
    print(f"\n📋 {text}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_info(text):
    print(f"ℹ️  {text}")

def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        print(e.stderr)
        return None

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print_error("Python 3.8+ required")
        sys.exit(1)
    print_success(f"Python {sys.version_info.major}.{sys.version_info.minor}")

def setup_venv():
    """Create and activate virtual environment"""
    venv_path = Path("openvino-env")
    
    if not venv_path.exists():
        print_step("Creating virtual environment...")
        result = run_command("python -m venv openvino-env")
        if not result:
            sys.exit(1)
        print_success("Virtual environment created")
    else:
        print_success("Virtual environment already exists")

def install_dependencies():
    """Install required packages"""
    print_step("Installing dependencies...")
    
    # Use the virtual environment's pip
    pip_cmd = "openvino-env/bin/pip"
    
    packages = [
        "openvino>=2024.0.0",
        "openvino-tokenizers>=2024.0.0", 
        "transformers>=4.40.0",
        "torch",
        "psutil",
        "regex"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = run_command(f"{pip_cmd} install {package}")
        if not result:
            print_error(f"Failed to install {package}")
            sys.exit(1)
    
    print_success("All dependencies installed")

def check_model():
    """Check if model exists and download if needed"""
    print_step("Checking for model...")
    
    model_path = Path("microsoft/DialoGPT-medium")
    if model_path.exists():
        print_success("Model found locally")
        return
    
    print("Model not found locally, will download on first run")
    print_info("This is normal - the model will be downloaded automatically when needed")

def run_test():
    """Run a simple test to verify everything works"""
    print_step("Running test...")
    
    # Create a simple test script
    test_code = '''
import sys
try:
    import openvino
    import transformers
    import torch
    import psutil
    import regex
    print("SUCCESS: All imports successful")
    
    # Test basic OpenVINO functionality
    core = openvino.Core()
    devices = core.available_devices
    print(f"SUCCESS: OpenVINO Core created, available devices: {devices}")
    
    print("SUCCESS: NPGlue is ready to use!")
    
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
    
    # Run the test using the virtual environment's Python
    python_cmd = "openvino-env/bin/python"
    result = run_command(f'{python_cmd} -c \'{test_code}\'')
    
    if result and result.returncode == 0:
        print_success("Test passed!")
        print(result.stdout)
        return True
    else:
        print_error("Test failed")
        if result:
            print(result.stderr)
        return False

def print_next_steps():
    """Tell user what to do next"""
    print_header("🎉 Setup Complete!")
    print_info("NPGlue is now installed and ready to use.")
    print("\n📝 Next steps:")
    print("1. Activate the environment:")
    print("   source openvino-env/bin/activate")
    print("\n2. Start the server:")
    print("   python server_production.py")
    print("\n3. Test the API:")
    print("   curl http://localhost:8080/health")
    print("\n4. Send a message:")
    print('   curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d \'{"message": "Hello!"}\'')

def main():
    print_header("NPGlue Simple Setup")
    print_info("This will install everything you need and run a test")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Run setup steps
    check_python()
    setup_venv()
    install_dependencies()
    check_model()
    
    # Run test
    if run_test():
        print_next_steps()
    else:
        print_error("Setup failed during testing")
        sys.exit(1)

if __name__ == "__main__":
    main()
