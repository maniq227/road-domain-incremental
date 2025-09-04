#!/usr/bin/env python3
"""
Main Pipeline Script for ROAD Continual Learning
===============================================

This script runs the complete pipeline for ROAD continual learning with RF-DETR:
1. Data preparation
2. Model setup
3. Avalanche setup
4. Training
5. Evaluation and visualization

Author: ROAD Continual Learning Project
Date: 2024
"""

import subprocess
import sys
import argparse
from pathlib import Path
import time

def run_script(script_path: str, description: str, args: list = None):
    """
    Run a Python script and handle errors.
    
    Args:
        script_path: Path to the script to run
        description: Description of what the script does
        args: Additional arguments to pass to the script
    """
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Avoid Unicode glyphs that may break on Windows cp1252
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] Error in {description}:")
        print(f"Return code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        return False

def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='Run complete ROAD continual learning pipeline')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='Skip data preparation step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--frames_per_video', type=int, default=50,
                       help='Number of frames to extract per video')
    
    args = parser.parse_args()
    
    print("ROAD Continual Learning Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Frames per video: {args.frames_per_video}")
    
    start_time = time.time()
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        success = run_script(
            "scripts/01_data_preparation.py",
            "Data Preparation - Converting videos to COCO format",
            ["--frames_per_video", str(args.frames_per_video)]
        )
        if not success:
            print("Pipeline failed at data preparation step")
            return 1
    else:
        print("Skipping data preparation step")
    
    # Step 2: Model Setup
    success = run_script(
        "scripts/02_model_setup.py",
        "Model Setup - Initializing RF-DETR with pretrained weights"
    )
    if not success:
        print("Pipeline failed at model setup step")
        return 1
    
    # Step 3: Avalanche Setup
    success = run_script(
        "scripts/03_avalanche_setup.py",
        "Avalanche Setup - Creating continual learning benchmark"
    )
    if not success:
        print("Pipeline failed at Avalanche setup step")
        return 1
    
    # Step 4: Training
    if not args.skip_training:
        success = run_script(
            "scripts/04_train_model.py",
            "Training - Running continual learning with naïve fine-tuning"
        )
        if not success:
            print("Pipeline failed at training step")
            return 1
    else:
        print("Skipping training step")
    
    # Step 5: Evaluation and Visualization
    success = run_script(
        "scripts/05_evaluate_and_visualize.py",
        "Evaluation - Computing metrics and generating visualizations"
    )
    if not success:
        print("Pipeline failed at evaluation step")
        return 1
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("- outputs/models/: Trained model checkpoints")
    print("- outputs/results/: Evaluation metrics and results")
    print("- outputs/visualizations/: Detection visualizations")
    print("- outputs/training_log.txt: Training logs")
    print("- outputs/continual_learning_metrics.json: BWT/FWT metrics")
    print("- outputs/performance_metrics.png: Performance plots")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
