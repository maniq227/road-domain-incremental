#!/usr/bin/env python3
"""
Training Script for ROAD Continual Learning with RF-DETR
========================================================

This script implements the main training pipeline using Avalanche for
domain-incremental continual learning on the ROAD dataset.

Author: ROAD Continual Learning Project
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Avalanche imports
import avalanche
from avalanche.benchmarks import GenericCLScenario
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
# MetricPlugin import not needed for this setup
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, forward_transfer_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent))
from model_setup import RFDETRModel, ROADModelManager
from avalanche_setup import ROADAvalancheSetup, ROADDataset

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

class DetectionLoss(nn.Module):
    """
    Custom loss function for object detection.
    Combines classification and bounding box regression losses.
    """
    
    def __init__(self, num_classes: int, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize detection loss.
        
        Args:
            num_classes: Number of object classes
            alpha: Weight for classification loss
            beta: Weight for regression loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        # Ignore index -1 in padded labels
        self.classification_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.regression_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions: Dict, targets: Any) -> torch.Tensor:
        """
        Compute detection loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of loss components
        """
        pred_logits = predictions['pred_logits']  # (B, num_queries, num_classes)
        pred_boxes = predictions['pred_boxes']    # (B, num_queries, 4)
        
        batch_size = pred_logits.size(0)
        device = pred_logits.device
        
        # Initialize losses
        total_class_loss = 0.0
        total_reg_loss = 0.0
        num_valid_samples = 0
        
        for b in range(batch_size):
            # Support both dict targets (eval) and packed tensor targets (train)
            if isinstance(targets, dict):
                num_b = targets['num_boxes'][b].item() if 'num_boxes' in targets else targets['boxes'][b].shape[0]
                target_boxes = targets['boxes'][b][:num_b]
                target_labels = targets['labels'][b][:num_b]
            else:
                # Packed tensor with shape (B, max_objects, 5): [x1,y1,x2,y2,label]
                # Filter out padded entries with label == -1
                packed_b = targets[b]
                labels_vec = packed_b[:, 4].long()
                valid_mask = labels_vec != -1
                target_boxes = packed_b[valid_mask, :4]
                target_labels = labels_vec[valid_mask]
            
            if target_boxes.numel() == 0:
                # No objects in this image - assign all predictions to background
                class_loss = self.classification_loss(
                    pred_logits[b].view(-1, self.num_classes),
                    torch.zeros(pred_logits.size(1), dtype=torch.long, device=device)
                )
                total_class_loss += class_loss
                num_valid_samples += 1
                continue
            
            # Simple assignment: match predictions to targets based on IoU
            # This is a simplified version - in practice, you'd use Hungarian algorithm
            num_queries = pred_logits.size(1)
            num_targets = len(target_boxes)
            
            # Create assignment matrix (simplified)
            assigned_queries = torch.zeros(num_queries, dtype=torch.bool, device=device)
            assigned_targets = torch.zeros(num_targets, dtype=torch.bool, device=device)
            
            # Simple greedy assignment
            for t in range(min(num_queries, num_targets)):
                if not assigned_targets[t]:
                    # Find best unassigned query
                    best_query = -1
                    best_iou = 0.0
                    
                    for q in range(num_queries):
                        if not assigned_queries[q]:
                            # Compute IoU (simplified)
                            pred_box = pred_boxes[b, q]
                            target_box = target_boxes[t]
                            
                            # Convert to [x1, y1, x2, y2] format
                            pred_x1, pred_y1, pred_x2, pred_y2 = pred_box
                            target_x1, target_y1, target_x2, target_y2 = target_box
                            
                            # Compute IoU
                            iou = self._compute_iou(
                                pred_x1, pred_y1, pred_x2, pred_y2,
                                target_x1, target_y1, target_x2, target_y2
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_query = q
                    
                    if best_query >= 0 and best_iou > 0.1:  # Threshold for assignment
                        # Assign query to target
                        assigned_queries[best_query] = True
                        assigned_targets[t] = True
                        
                        # Classification loss
                        class_loss = self.classification_loss(
                            pred_logits[b, best_query:best_query+1],
                            target_labels[t:t+1]
                        )
                        total_class_loss += class_loss
                        
                        # Regression loss
                        reg_loss = self.regression_loss(
                            pred_boxes[b, best_query],
                            target_boxes[t]
                        )
                        total_reg_loss += reg_loss
            
            # Background loss for unassigned queries
            for q in range(num_queries):
                if not assigned_queries[q]:
                    class_loss = self.classification_loss(
                        pred_logits[b, q:q+1],
                        torch.zeros(1, dtype=torch.long, device=device)
                    )
                    total_class_loss += class_loss
            
            num_valid_samples += 1
        
        # Average losses
        if num_valid_samples > 0:
            avg_class_loss = total_class_loss / num_valid_samples
            avg_reg_loss = total_reg_loss / num_valid_samples
        else:
            avg_class_loss = torch.tensor(0.0, device=device)
            avg_reg_loss = torch.tensor(0.0, device=device)
        
        total_loss = self.alpha * avg_class_loss + self.beta * avg_reg_loss
        
        return total_loss
    
    def _compute_iou(self, x1, y1, x2, y2, x1_gt, y1_gt, x2_gt, y2_gt):
        """Compute IoU between two bounding boxes."""
        # Intersection
        inter_x1 = torch.max(x1, x1_gt)
        inter_y1 = torch.max(y1, y1_gt)
        inter_x2 = torch.min(x2, x2_gt)
        inter_y2 = torch.min(y2, y2_gt)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou

class ROADContinualTrainer:
    """Main trainer class for ROAD continual learning."""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing processed datasets
            output_dir: Output directory for results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model_manager = ROADModelManager(str(self.output_dir / 'models'))
        self.avalanche_setup = ROADAvalancheSetup(str(self.data_dir), str(self.output_dir))
        
        # Training history
        self.training_history = {
            'experiences': [],
            'metrics': [],
            'timestamps': []
        }
    
    def setup_training(self):
        """Set up training components."""
        print("Setting up training components...")
        
        # Load domain datasets
        domain_datasets = self.avalanche_setup.load_domain_datasets()
        
        if not domain_datasets:
            raise ValueError("No domain datasets found!")
        
        # Create benchmark
        self.benchmark = self.avalanche_setup.create_avalanche_benchmark(domain_datasets)
        
        # Create model
        self.model = self.model_manager.create_model(pretrained=True)
        
        # Set up loggers
        self.loggers = self.avalanche_setup.setup_loggers()
        
        # Set up evaluation plugin
        self.eval_plugin = self.avalanche_setup.setup_evaluation_plugin(self.loggers)
        
        # Create custom loss function
        self.criterion = DetectionLoss(num_classes=len(ROAD_CLASSES))
        
        # Create strategy
        self.strategy = self._create_strategy()
        
        print("[OK] Training setup complete")
    
    def _create_strategy(self) -> Naive:
        """Create training strategy."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        
        # Creates the naive strategy
        strategy = Naive(
            model=self.model,
            optimizer=optimizer,
            criterion=self.criterion,
            train_mb_size=2,
            train_epochs=3,
            eval_mb_size=2,
            device=self.device,
            evaluator=self.eval_plugin,
            plugins=[LRSchedulerPlugin(scheduler)]
        )
        
        return strategy
    
    def train_continual(self):
        """Run continual learning training."""
        print("Starting continual learning training...")
        # Get train stream from benchmark  
        train_stream = self.benchmark.train_datasets_stream
        print(f"Number of experiences: {len(train_stream)}")
        
        start_time = time.time()
        experience_times = []
        
        for experience_id, experience in enumerate(train_stream):
            print(f"\n{'='*60}")
            print(f"EXPERIENCE {experience_id + 1}/{len(self.avalanche_setup.domains)}")
            print(f"Domain: {self.avalanche_setup.domains[experience_id]}")
            print(f"Training samples: {len(experience.dataset)}")
            print(f"{'='*60}")
            if experience_times:
                avg_time = sum(experience_times) / len(experience_times)
                remaining = len(self.avalanche_setup.domains) - experience_id
                eta_secs = avg_time * remaining
                print(f"Estimated time remaining (experiences): {eta_secs/60:.1f} min")
            
            # Train on current experience (naive finetuning)
            exp_start = time.time()
            self.strategy.train(experience)
            
            # Save checkpoint after each domain
            checkpoint_name = f"model_after_domain_{experience_id + 1}.pth"
            self.model_manager.save_model(
                self.model,
                checkpoint_name,
                {
                    'experience_id': experience_id,
                    'domain': self.avalanche_setup.domains[experience_id],
                    'timestamp': time.time()
                }
            )

            # Skip eval during training to prevent hanging - use separate eval script
            print("Skipping eval during training (will run separate evaluation)")
            
            # Record training info
            self.training_history['experiences'].append(experience_id)
            self.training_history['timestamps'].append(time.time())
            
            exp_time = time.time() - exp_start
            experience_times.append(exp_time)
            print(f"[OK] Experience {experience_id + 1} completed in {exp_time/60:.1f} min")
            done = experience_id + 1
            total = len(self.avalanche_setup.domains)
            elapsed = time.time() - start_time
            avg_time = sum(experience_times) / len(experience_times)
            remaining_secs = avg_time * (total - done)
            progress_pct = 100.0 * done / total
            print(f"Progress: {done}/{total} experiences ({progress_pct:.1f}%) | Elapsed: {elapsed/60:.1f} min | ETA: {remaining_secs/60:.1f} min")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"{'='*60}")
        
        # Save training history
        self._save_training_history()
    
    def _save_training_history(self):
        """Save training history."""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
    
    def compute_continual_metrics(self):
        """Compute continual learning metrics (BWT, FWT)."""
        print("Computing continual learning metrics...")
        
        # This would typically be done using Avalanche's built-in metrics
        # For now, we'll create a placeholder structure
        
        metrics = {
            'backward_transfer': {},
            'forward_transfer': {},
            'final_accuracy': {},
            'average_accuracy': 0.0
        }
        
        # Save metrics
        metrics_path = self.output_dir / 'continual_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Continual learning metrics saved to: {metrics_path}")
        return metrics

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train RF-DETR on ROAD dataset with continual learning')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing processed ROAD datasets')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ROADContinualTrainer(data_dir=args.data_dir, output_dir=args.output_dir)
    
    try:
        # Set up training
        trainer.setup_training()
        
        # Run training
        trainer.train_continual()
        
        # Compute metrics
        trainer.compute_continual_metrics()
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        print("Check the following files:")
        print("- outputs/models/: Trained model checkpoints")
        print("- outputs/training_log.txt: Training logs")
        print("- outputs/continual_metrics.json: Continual learning metrics")
        print("- outputs/tensorboard_logs/: TensorBoard logs")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
