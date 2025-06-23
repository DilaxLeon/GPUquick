#!/usr/bin/env python3
"""
Quick setup test for GPU QuickCap
Run this to verify your installation is correct
"""

import os
import sys
from pathlib import Path

def test_setup():
    print("🔍 Testing GPU QuickCap Setup...")
    print("=" * 50)
    
    # Test 1: Check required directories
    print("📁 Checking directories...")
    required_dirs = ["uploads", "captions", "templates"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/ exists")
        else:
            print(f"  ❌ {dir_name}/ missing - creating...")
            os.makedirs(dir_name, exist_ok=True)
    
    # Test 2: Check template file
    print("\n🎨 Checking template...")
    if os.path.exists("templates/index.html"):
        print("  ✅ templates/index.html exists")
    else:
        print("  ❌ templates/index.html missing")
    
    # Test 3: Check Python dependencies
    print("\n📦 Checking Python dependencies...")
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("whisper", "OpenAI Whisper"),
        ("PIL", "Pillow"),
        ("jinja2", "Jinja2")
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} installed")
        except ImportError:
            print(f"  ❌ {name} missing - install with: pip install {package}")
    
    # Test 4: Check FFmpeg
    print("\n🎬 Checking FFmpeg...")
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✅ FFmpeg is available")
            # Check for CUDA support
            if "cuda" in result.stdout.lower():
                print("  ✅ CUDA support detected in FFmpeg")
            else:
                print("  ⚠️  CUDA support not detected - GPU acceleration may not work")
        else:
            print("  ❌ FFmpeg not working properly")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ❌ FFmpeg not found in PATH")
        print("      Download from: https://ffmpeg.org/download.html")
    
    # Test 5: Check CUDA/GPU
    print("\n🚀 Checking GPU acceleration...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available - GPU: {gpu_name}")
        else:
            print("  ⚠️  CUDA not available - will use CPU (slower)")
    except ImportError:
        print("  ⚠️  PyTorch not installed - GPU detection skipped")
    
    # Test 6: Check font
    print("\n🔤 Checking font...")
    font_paths = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/arial.ttf"
    ]
    
    font_found = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            print(f"  ✅ Font found: {font_path}")
            font_found = True
            break
    
    if not font_found:
        print("  ⚠️  Arial font not found - you may need to:")
        print("      - Install Arial font on your system")
        print("      - Or modify FONT_PATH in app.py to use a different font")
    
    print("\n" + "=" * 50)
    print("🎯 Setup test complete!")
    print("\n🚀 To start the server, run:")
    print("   python run.py")
    print("\n🌐 Then open: http://localhost:8000")

if __name__ == "__main__":
    test_setup()