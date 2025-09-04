#!/usr/bin/env python3
"""
Avalanche Setup Script for ROAD Continual Learning
=================================================

This script sets up the Avalanche framework for domain-incremental continual learning
on the ROAD dataset with 4 weather domains.

Author: ROAD Continual Learning Project
Date: 2024
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict

# Avalanche imports
import avalanche
from avalanche.benchmarks import GenericCLScenario
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
# MetricPlugin import not needed for this setup
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, forward_transfer_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.benchmarks import benchmark_from_datasets

# ROAD dataset class mappings
ROAD_CLASSES = {
    'Car': 0,
    'Bus': 1, 
    'Truck': 2,
    'Motorcycle': 3,
    'Bicycle': 4,
    'Pedestrian': 5,
    'Van': 6,
    'Trailer': 7,
    'Emergency_vehicle': 8,
    'Other_vehicle': 9,
    'Other_agent': 10
}

class ROADDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for ROAD data in COCO format.
    """
    
    def __init__(self, images_dir: str, annotations_file: str, transform=None, return_packed: bool = False):
        """
        Initialize ROAD dataset.
        
        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format annotations JSON
            transform: Image transforms
            return_packed: If True, returns (image, packed_tensor) where packed is (max_objects, 5)
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.return_packed = return_packed
        # Maximum number of objects per image for padding (avoid variable-size collation issues)
        self.max_objects = 50
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = defaultdict(list)
        
        for ann in self.coco_data['annotations']:
            self.annotations[ann['image_id']].append(ann)
        
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        image_path = self.images_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations for this image
        anns = self.annotations[image_id]
        
        # Create target tensor
        boxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            labels.append(ann['category_id'])
        
        # Convert to tensors and pad to fixed size [max_objects]
        num_objects = len(boxes)
        if num_objects > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        # Pad
        padded_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        # Use -1 as sentinel for padded labels so loss can ignore
        padded_labels = torch.full((self.max_objects,), fill_value=-1, dtype=torch.long)
        if num_objects > 0:
            take = min(num_objects, self.max_objects)
            padded_boxes[:take] = boxes_t[:take]
            padded_labels[:take] = labels_t[:take]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.return_packed:
            # Pack into (max_objects, 5): [x1,y1,x2,y2,label]
            packed = torch.zeros((self.max_objects, 5), dtype=torch.float32)
            packed[:, :4] = padded_boxes
            packed[:, 4] = padded_labels.to(torch.float32)
            return image, packed
        else:
            return image, {
                'boxes': padded_boxes,           # (max_objects, 4)
                'labels': padded_labels,         # (max_objects,)
                'num_boxes': torch.tensor(min(num_objects, self.max_objects), dtype=torch.long),
                'image_id': image_id,
                'domain': image_info.get('domain', 'unknown')
            }

class ROADAvalancheSetup:
    """Sets up Avalanche framework for ROAD continual learning."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        """
        Initialize Avalanche setup.
        
        Args:
            data_dir: Directory containing processed ROAD datasets
            output_dir: Output directory for logs and results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Weather domains
        self.domains = ['sunny', 'overcast', 'snowy', 'night']
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_domain_datasets(self) -> Dict[str, Dict[str, ROADDataset]]:
        """
        Load datasets for all domains.
        
        Returns:
            Dictionary mapping domain names to train/test datasets
        """
        domain_datasets = {}
        
        for domain in self.domains:
            domain_dir = self.data_dir / domain
            
            if not domain_dir.exists():
                print(f"Warning: Domain directory not found: {domain_dir}")
                continue
            
            # Load train dataset
            train_images_dir = domain_dir / 'images'
            train_annotations = domain_dir / 'annotations' / 'train.json'
            
            if train_annotations.exists():
                train_dataset = ROADDataset(
                    images_dir=str(train_images_dir),
                    annotations_file=str(train_annotations),
                    transform=self.train_transform,
                    return_packed=True
                )
            else:
                print(f"Warning: Train annotations not found for {domain}")
                continue
            
            # Load test dataset
            test_annotations = domain_dir / 'annotations' / 'test.json'
            
            if test_annotations.exists():
                test_dataset = ROADDataset(
                    images_dir=str(train_images_dir),  # Same images directory
                    annotations_file=str(test_annotations),
                    transform=self.test_transform,
                    return_packed=False
                )
            else:
                print(f"Warning: Test annotations not found for {domain}")
                continue
            
            domain_datasets[domain] = {
                'train': train_dataset,
                'test': test_dataset
            }
            
            print(f"[OK] Loaded {domain}: {len(train_dataset)} train, {len(test_dataset)} test samples")
        
        return domain_datasets
    
    def create_avalanche_benchmark(self, domain_datasets: Dict[str, Dict[str, ROADDataset]]) -> GenericCLScenario:
        print("Creating Avalanche benchmark...")

        train_datasets = []
        test_datasets = []

        for domain in self.domains:
            if domain not in domain_datasets:
                print(f"Warning: Skipping domain {domain} - no data found")
                continue

            train_datasets.append(AvalancheDataset(domain_datasets[domain]['train']))
            test_datasets.append(AvalancheDataset(domain_datasets[domain]['test']))

        benchmark = benchmark_from_datasets(
            train_datasets=train_datasets,
            test_datasets=test_datasets
        )

        print(f"[OK] Created benchmark with {len(train_datasets)} experiences")
        return benchmark
    
    def setup_loggers(self) -> List:
        """
        Set up logging for training.
        
        Returns:
            List of loggers
        """
        loggers = []
        
        # Interactive logger
        loggers.append(InteractiveLogger())
        
        # Text logger
        text_logger = TextLogger(open(self.output_dir / 'training_log.txt', 'w'))
        loggers.append(text_logger)
        
        # Tensorboard logger
        tensorboard_logger = TensorboardLogger(
            tb_log_dir=self.output_dir / 'tensorboard_logs'
        )
        loggers.append(tensorboard_logger)
        
        return loggers
    
    def setup_evaluation_plugin(self, loggers: List) -> EvaluationPlugin:
        metrics = [
            loss_metrics(epoch=True, experience=True, stream=True)
        ]
        eval_plugin = EvaluationPlugin(
            *metrics,
            loggers=loggers
        )
        return eval_plugin
    
    def create_naive_strategy(self, model: nn.Module, eval_plugin: EvaluationPlugin) -> Naive:
        """
        Create Naive continual learning strategy.
        
        Args:
            model: Model to train
            eval_plugin: Evaluation plugin
            
        Returns:
            Naive strategy
        """
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        
        # Create strategy
        strategy = Naive(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=8,
            train_epochs=10,
            eval_mb_size=8,
            device=self.device,
            evaluator=eval_plugin,
            plugins=[LRSchedulerPlugin(scheduler)]
        )
        
        return strategy
    
    def save_benchmark_info(self, benchmark: GenericCLScenario):
        """
        Save benchmark information.
        
        Args:
            benchmark: Avalanche benchmark
        """
        benchmark_info = {
            'n_experiences': len(self.domains),
            'domains': self.domains,
            'classes': list(ROAD_CLASSES.keys()),
            'class_mapping': ROAD_CLASSES
        }
        
        info_path = self.output_dir / 'benchmark_info.json'
        with open(info_path, 'w') as f:
            json.dump(benchmark_info, f, indent=2)
        
        print(f"Benchmark info saved to: {info_path}")

def main():
    """Main function to set up Avalanche."""
    parser = argparse.ArgumentParser(description='Set up Avalanche for ROAD continual learning')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing processed ROAD datasets')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for logs and results')
    
    args = parser.parse_args()
    
    # Create Avalanche setup
    setup = ROADAvalancheSetup(data_dir=args.data_dir, output_dir=args.output_dir)
    
    # Load domain datasets
    print("Loading domain datasets...")
    domain_datasets = setup.load_domain_datasets()
    
    if not domain_datasets:
        print("Error: No domain datasets found!")
        return
    
    # Create benchmark
    benchmark = setup.create_avalanche_benchmark(domain_datasets)
    
    # Save benchmark info
    setup.save_benchmark_info(benchmark)
    
    # Set up loggers
    loggers = setup.setup_loggers()
    
    # Set up evaluation plugin
    eval_plugin = setup.setup_evaluation_plugin(loggers)
    
    print("\n" + "="*50)
    print("AVALANCHE SETUP COMPLETE")
    print("="*50)
    print(f"Benchmark created with {len(setup.domains)} experiences")
    print(f"Domains: {setup.domains}")
    print(f"Classes: {list(ROAD_CLASSES.keys())}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {setup.device}")

if __name__ == "__main__":
    main()
