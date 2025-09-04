# ROAD Continual Learning Pipeline Guide

This guide provides detailed instructions for running the ROAD continual learning pipeline with RF-DETR and Avalanche.

## Overview

The pipeline implements domain-incremental continual learning on the ROAD benchmark dataset, training RF-DETR sequentially across 4 weather domains (sunny, overcast, snowy, night) using naïve fine-tuning.

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 10GB+ disk space

### Required Packages
All dependencies are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **PyTorch**: Deep learning framework
- **Avalanche**: Continual learning framework
- **OpenCV**: Video processing
- **timm**: Pre-trained models
- **Matplotlib/Seaborn**: Visualization

## Dataset Structure

The ROAD dataset contains 4 video files representing different weather conditions:

```
videos/
├── 2015-02-06-13-57-16_stereo_centre_01.mp4  # Sunny
├── 2014-06-26-09-31-18_stereo_centre_02.mp4  # Overcast
├── 2015-02-03-08-45-10_stereo_centre_04.mp4  # Snowy
└── 2014-12-10-18-10-50_stereo_centre_02.mp4  # Night
```

## Pipeline Steps

### 1. Data Preparation (`01_data_preparation.py`)

Converts ROAD videos into COCO format image datasets.

**Features:**
- Extracts frames from videos (configurable number per video)
- Creates synthetic annotations for 11 active agent classes
- Splits data into train/test sets (80/20)
- Generates COCO format JSON files

**Usage:**
```bash
python scripts/01_data_preparation.py --frames_per_video 100
```

**Output:**
```
data/
├── sunny/
│   ├── images/
│   └── annotations/
│       ├── train.json
│       └── test.json
├── overcast/
├── snowy/
└── night/
```

### 2. Model Setup (`02_model_setup.py`)

Initializes RF-DETR model with pretrained weights.

**Features:**
- Creates RF-DETR model architecture
- Loads COCO pretrained weights
- Adapts for 11 ROAD classes
- Saves initial model checkpoint

**Usage:**
```bash
python scripts/02_model_setup.py
```

**Output:**
```
outputs/models/
├── initial_model.pth
└── model_config.json
```

### 3. Avalanche Setup (`03_avalanche_setup.py`)

Sets up Avalanche framework for continual learning.

**Features:**
- Creates domain-incremental benchmark
- Defines 4 experiences (one per domain)
- Sets up logging and evaluation
- Configures data loaders

**Usage:**
```bash
python scripts/03_avalanche_setup.py
```

**Output:**
```
outputs/
├── benchmark_info.json
└── training_log.txt
```

### 4. Training (`04_train_model.py`)

Runs continual learning training with naïve fine-tuning.

**Features:**
- Sequential training across domains
- Custom detection loss function
- Model checkpointing after each domain
- Real-time evaluation

**Usage:**
```bash
python scripts/04_train_model.py
```

**Output:**
```
outputs/models/
├── model_after_domain_1.pth
├── model_after_domain_2.pth
├── model_after_domain_3.pth
└── model_after_domain_4.pth
```

### 5. Evaluation (`05_evaluate_and_visualize.py`)

Evaluates model performance and generates visualizations.

**Features:**
- Computes mAP metrics for each domain
- Calculates BWT and FWT metrics
- Generates detection visualizations
- Creates performance plots

**Usage:**
```bash
python scripts/05_evaluate_and_visualize.py
```

**Output:**
```
outputs/
├── evaluation_results.json
├── continual_learning_metrics.json
├── performance_metrics.png
└── visualizations/
    ├── sunny/
    ├── overcast/
    ├── snowy/
    └── night/
```

## Quick Start

### Option 1: Run Complete Pipeline
```bash
python run_pipeline.py
```

### Option 2: Run Individual Steps
```bash
# Test setup first
python test_setup.py

# Run each step individually
python scripts/01_data_preparation.py
python scripts/02_model_setup.py
python scripts/03_avalanche_setup.py
python scripts/04_train_model.py
python scripts/05_evaluate_and_visualize.py
```

### Option 3: Skip Steps
```bash
# Skip data preparation if already done
python run_pipeline.py --skip_data_prep

# Skip training if already done
python run_pipeline.py --skip_training
```

## Configuration

### Data Preparation
- `--frames_per_video`: Number of frames to extract per video (default: 100)
- `--video_dir`: Directory containing video files (default: videos)
- `--output_dir`: Output directory for processed data (default: data)

### Training
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of epochs per domain (default: 5)
- `--learning_rate`: Learning rate (default: 1e-4)

### Model
- `--num_classes`: Number of object classes (default: 11)
- `--input_size`: Input image size (default: 640x640)

## Expected Results

### Performance Metrics
- **mAP**: Mean Average Precision for object detection
- **BWT**: Backward Transfer (forgetting measure)
- **FWT**: Forward Transfer (knowledge transfer measure)

### Typical Results
- mAP scores: 0.3-0.7 depending on domain complexity
- BWT: Negative values indicate forgetting
- FWT: Positive values indicate positive transfer

### Output Files
- Model checkpoints after each domain
- Training logs and metrics
- Detection visualizations
- Performance plots

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training script
   - Use smaller input image size
   - Process fewer frames per video

2. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check CUDA installation for GPU support

3. **Video Processing Errors**
   - Verify video files are not corrupted
   - Check OpenCV installation
   - Ensure sufficient disk space

4. **Model Loading Issues**
   - Check model checkpoint paths
   - Verify model architecture compatibility
   - Ensure proper device configuration

### Debug Mode
Run individual scripts with verbose output:
```bash
python scripts/01_data_preparation.py --verbose
```

## Advanced Usage

### Custom Model Architecture
Modify `scripts/02_model_setup.py` to use different backbones or architectures.

### Custom Loss Functions
Update the detection loss in `scripts/04_train_model.py` for different objectives.

### Additional Metrics
Extend evaluation in `scripts/05_evaluate_and_visualize.py` for custom metrics.

### Memory Management
Implement experience replay or regularization techniques in the training script.

## Citation

If you use this code, please cite:
- ROAD: The ROad event Awareness Dataset for autonomous driving
- RF-DETR: Real-time Detection Transformer
- Avalanche: An End-to-End Library for Continual Learning

## License

This project is licensed under the MIT License - see the LICENSE file for details.
