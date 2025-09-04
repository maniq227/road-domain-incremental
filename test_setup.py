#!/usr/bin/env python3
"""
Test Setup Script for ROAD Continual Learning
============================================

This script tests the setup and verifies that all components are working correctly.

Author: ROAD Continual Learning Project
Date: 2024
"""

import torch
import sys
from pathlib import Path
import importlib.util

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'avalanche',
        'cv2',
        'PIL',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
        'timm'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("✓ All packages imported successfully")
    return True

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ PyTorch version: {torch.__version__}")
        return True
    else:
        print("⚠ CUDA not available - will use CPU")
        return False

def test_avalanche():
    """Test Avalanche installation."""
    print("\nTesting Avalanche...")
    
    try:
        import avalanche
        print(f"✓ Avalanche version: {avalanche.__version__}")
        
        # Test basic Avalanche functionality
        from avalanche.benchmarks import GenericCLScenario
        from avalanche.training import Naive
        print("✓ Avalanche components imported successfully")
        return True
    except Exception as e:
        print(f"✗ Avalanche test failed: {e}")
        return False

def test_custom_modules():
    """Test custom module imports."""
    print("\nTesting custom modules...")
    
    scripts_dir = Path("scripts")
    if not scripts_dir.exists():
        print("✗ Scripts directory not found")
        return False
    
    # Test importing our custom modules
    try:
        sys.path.append(str(scripts_dir))
        
        # Test model setup
        from model_setup import RFDETRModel, ROADModelManager
        print("✓ Model setup module imported")
        
        # Test Avalanche setup
        from avalanche_setup import ROADAvalancheSetup, ROADDataset
        print("✓ Avalanche setup module imported")
        
        return True
    except Exception as e:
        print(f"✗ Custom module import failed: {e}")
        return False

def test_data_structure():
    """Test data directory structure."""
    print("\nTesting data structure...")
    
    required_dirs = ['data', 'scripts', 'outputs', 'videos']
    required_files = ['requirements.txt', 'README.md']
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            return False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            return False
    
    # Check video files
    videos_dir = Path("videos")
    video_files = list(videos_dir.glob("*.mp4"))
    
    if len(video_files) >= 4:
        print(f"✓ Found {len(video_files)} video files")
        for video in video_files:
            print(f"  - {video.name}")
    else:
        print(f"✗ Expected 4 video files, found {len(video_files)}")
        return False
    
    return True

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        sys.path.append("scripts")
        from model_setup import RFDETRModel
        
        # Create a small test model
        model = RFDETRModel(num_classes=11, pretrained=False)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        print("✓ Model created successfully")
        print(f"✓ Forward pass successful: {output['pred_logits'].shape}, {output['pred_boxes'].shape}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ROAD Continual Learning Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Availability", test_cuda),
        ("Avalanche Installation", test_avalanche),
        ("Custom Modules", test_custom_modules),
        ("Data Structure", test_data_structure),
        ("Model Creation", test_model_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is ready.")
        print("\nYou can now run the pipeline with:")
        print("python run_pipeline.py")
    else:
        print("⚠ Some tests failed. Please fix the issues before running the pipeline.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
