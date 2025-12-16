#!/usr/bin/env python3
"""
Quick script to check Sophon SDK installation and BModel availability.
Run this before deploying to verify your Sophon setup.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.sophon_utils import print_sophon_status, find_bmodel, is_sophon_available
from config.settings import Config


def main():
    print("\n" + "=" * 60)
    print("CuTrack - Sophon Deployment Check")
    print("=" * 60 + "\n")
    
    # Check Sophon SDK
    print_sophon_status()
    
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)
    
    # Check model directory
    models_dir = Config.MODELS_DIR
    print(f"\nModels directory: {models_dir}")
    if os.path.exists(models_dir):
        print(f"✅ Models directory exists")
        files = [f for f in os.listdir(models_dir) if f.endswith('.bmodel')]
        if files:
            print(f"   Found {len(files)} BModel file(s):")
            for f in files:
                print(f"     - {f}")
        else:
            print("   ⚠️  No .bmodel files found")
    else:
        print(f"⚠️  Models directory does not exist (will be created)")
        os.makedirs(models_dir, exist_ok=True)
    
    # Check output directories
    print(f"\nOutput directories:")
    for dir_name, dir_path in [
        ("Output", Config.OUTPUT_DIR),
        ("Videos", Config.VIDEO_OUTPUT_DIR),
        ("Reports", Config.REPORT_OUTPUT_DIR),
    ]:
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "⚠️"
        print(f"   {status} {dir_name}: {dir_path}")
    
    # Final recommendation
    print("\n" + "=" * 60)
    print("Recommendations")
    print("=" * 60)
    
    sophon_ok = is_sophon_available()
    bmodel_ok = find_bmodel() is not None
    
    if sophon_ok and bmodel_ok:
        print("✅ Everything looks good! You're ready to deploy.")
        print("\nTo start using CuTrack with Sophon:")
        print("   python3 main.py")
    elif sophon_ok and not bmodel_ok:
        print("⚠️  Sophon SDK is installed, but no BModel found.")
        print("\nNext steps:")
        print("1. Convert your YOLOv8 model to BModel (see README_SOPHON_SE9.md)")
        print(f"2. Place the .bmodel file in: {models_dir}/")
        print("3. Run this check again")
    elif not sophon_ok:
        print("⚠️  Sophon SDK is not available.")
        print("\nThe system will fall back to PyTorch/YOLO if available.")
        print("For optimal performance on Sophon devices, install Sophon SDK.")
        print("\nSee README_SOPHON_SE9.md for installation instructions.")
    
    print()


if __name__ == "__main__":
    main()

