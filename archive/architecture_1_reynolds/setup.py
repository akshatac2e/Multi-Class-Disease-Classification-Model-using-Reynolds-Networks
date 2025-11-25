"""
Setup script for Blood Cell Analysis System
===========================================
Run this script to verify your environment and setup.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    required = [
        'tensorflow',
        'numpy',
        'opencv-python',
        'scipy',
        'scikit-learn',
        'Pillow'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_').replace('opencv_python', 'cv2'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_directory_structure():
    """Check if required directories exist"""
    print("\nChecking directory structure...")
    required_dirs = [
        'config',
        'data',
        'data/rbc_data',
        'data/wbc_data',
        'data/segmentation',
        'models',
        'checkpoints',
        'results'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/")
            missing.append(dir_path)
    
    if missing:
        print(f"\n⚠️  Creating missing directories...")
        for dir_path in missing:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}/")
    
    return True

def check_gpu():
    """Check if GPU is available"""
    print("\nChecking GPU availability...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ {len(gpus)} GPU(s) detected:")
            for gpu in gpus:
                print(f"   - {gpu.name}")
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
    
    return True

def verify_files():
    """Verify that core files exist"""
    print("\nVerifying core files...")
    required_files = [
        'blood_cell_system.py',
        'training_pipeline.py',
        'inference_pipeline.py',
        'example_usage.py',
        'requirements.txt',
        'config/system_config.yaml'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing core files: {', '.join(missing)}")
        return False
    
    return True

def main():
    """Run all checks"""
    print("="*80)
    print("BLOOD CELL ANALYSIS SYSTEM - SETUP VERIFICATION")
    print("="*80)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Directory Structure", check_directory_structure),
        ("Core Files", verify_files),
        ("GPU", check_gpu)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "="*80)
    print("SETUP VERIFICATION SUMMARY")
    print("="*80)
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✅ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Place your data in data/rbc_data/ and data/wbc_data/")
        print("  2. Run: python example_usage.py")
        print("  3. See README_SYSTEM.md for detailed documentation")
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
        print("\nTo fix common issues:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: python --version")
        print("  - Ensure all core files are present")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
