#!/usr/bin/env python3
"""
Evaluation and Visualization Script for ROAD Continual Learning
==============================================================

This script evaluates the trained RF-DETR model and generates visualizations
including detection results, performance metrics, and continual learning analysis.

Author: ROAD Continual Learning Project
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm

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

# Color palette for visualization
CLASS_COLORS = [
    (255, 0, 0),     # Car - Red
    (0, 255, 0),     # Bus - Green
    (0, 0, 255),     # Truck - Blue
    (255, 255, 0),   # Motorcycle - Yellow
    (255, 0, 255),   # Bicycle - Magenta
    (0, 255, 255),   # Pedestrian - Cyan
    (128, 0, 128),   # Van - Purple
    (255, 165, 0),   # Trailer - Orange
    (0, 128, 0),     # Emergency_vehicle - Dark Green
    (128, 128, 128), # Other_vehicle - Gray
    (255, 192, 203)  # Other_agent - Pink
]

class ROADEvaluator:
    """Evaluates RF-DETR model on ROAD dataset."""
    
    def __init__(self, model_dir: str, data_dir: str, output_dir: str):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Directory containing trained models
            data_dir: Directory containing test datasets
            output_dir: Output directory for results
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model_manager = ROADModelManager(str(self.model_dir))
        self.avalanche_setup = ROADAvalancheSetup(str(self.data_dir), str(self.output_dir))
        
        # Results storage
        self.evaluation_results = {}
    
    def load_model(self, checkpoint_name: str) -> RFDETRModel:
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint file
            
        Returns:
            Loaded model
        """
        model = self.model_manager.load_model(checkpoint_name)
        model.eval()
        return model
    
    def compute_map_metrics(self, model: RFDETRModel, test_dataset: ROADDataset, domain: str) -> Dict[str, float]:
        """
        Compute mAP (mean Average Precision) metrics.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            domain: Domain name
            
        Returns:
            Dictionary of mAP metrics
        """
        print(f"Computing mAP for {domain} domain...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for idx in tqdm(range(len(test_dataset)), desc=f"Evaluating {domain}"):
                image, target = test_dataset[idx]
                image = image.unsqueeze(0).to(self.device)
                
                # Get predictions
                predictions = model(image)
                
                # Process predictions (simplified)
                pred_logits = predictions['pred_logits'][0]  # (num_queries, num_classes)
                pred_boxes = predictions['pred_boxes'][0]    # (num_queries, 4)
                
                # Convert to detection format
                pred_scores = torch.softmax(pred_logits, dim=-1)
                pred_classes = torch.argmax(pred_scores, dim=-1)
                pred_confidences = torch.max(pred_scores, dim=-1)[0]
                
                # Filter predictions by confidence
                valid_preds = pred_confidences > 0.5
                
                if valid_preds.sum() > 0:
                    valid_boxes = pred_boxes[valid_preds]
                    valid_classes = pred_classes[valid_preds]
                    valid_scores = pred_confidences[valid_preds]
                    
                    # Store predictions
                    all_predictions.append({
                        'boxes': valid_boxes.cpu().numpy(),
                        'classes': valid_classes.cpu().numpy(),
                        'scores': valid_scores.cpu().numpy()
                    })
                else:
                    all_predictions.append({
                        'boxes': np.array([]),
                        'classes': np.array([]),
                        'scores': np.array([])
                    })
                
                # Store targets
                all_targets.append({
                    'boxes': target['boxes'].numpy(),
                    'classes': target['labels'].numpy()
                })
        
        # Compute mAP (simplified version)
        map_metrics = self._compute_simplified_map(all_predictions, all_targets)
        
        return map_metrics
    
    def _compute_simplified_map(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """
        Compute simplified mAP metrics.
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
            
        Returns:
            Dictionary of mAP metrics
        """
        # This is a simplified mAP computation
        # In practice, you would use proper COCO evaluation metrics
        
        total_precision = 0.0
        total_recall = 0.0
        num_images = len(predictions)
        
        for pred, target in zip(predictions, targets):
            if len(pred['boxes']) == 0 and len(target['boxes']) == 0:
                # No objects in either prediction or target
                total_precision += 1.0
                total_recall += 1.0
            elif len(pred['boxes']) == 0:
                # No predictions but targets exist
                total_precision += 0.0
                total_recall += 0.0
            elif len(target['boxes']) == 0:
                # Predictions but no targets
                total_precision += 0.0
                total_recall += 1.0
            else:
                # Both predictions and targets exist
                # Simplified precision/recall computation
                num_preds = len(pred['boxes'])
                num_targets = len(target['boxes'])
                
                # Simple matching based on class
                pred_classes = set(pred['classes'])
                target_classes = set(target['classes'])
                
                correct_matches = len(pred_classes.intersection(target_classes))
                
                precision = correct_matches / num_preds if num_preds > 0 else 0.0
                recall = correct_matches / num_targets if num_targets > 0 else 0.0
                
                total_precision += precision
                total_recall += recall
        
        avg_precision = total_precision / num_images
        avg_recall = total_recall / num_images
        
        # Simplified mAP (average of precision and recall)
        map_score = (avg_precision + avg_recall) / 2.0
        
        return {
            'mAP': map_score,
            'precision': avg_precision,
            'recall': avg_recall
        }
    
    def visualize_detections(self, model: RFDETRModel, test_dataset: ROADDataset, 
                           domain: str, num_samples: int = 5):
        """
        Visualize detection results on sample images.
        
        Args:
            model: Trained model
            test_dataset: Test dataset
            domain: Domain name
            num_samples: Number of samples to visualize
        """
        print(f"Generating detection visualizations for {domain} domain...")
        
        model.eval()
        vis_dir = self.output_dir / 'visualizations' / domain
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Select random samples
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                image, target = test_dataset[idx]
                original_image = image.clone()
                
                # Normalize image for visualization
                image = image.unsqueeze(0).to(self.device)
                
                # Get predictions
                predictions = model(image)
                
                # Process predictions
                pred_logits = predictions['pred_logits'][0]
                pred_boxes = predictions['pred_boxes'][0]
                
                pred_scores = torch.softmax(pred_logits, dim=-1)
                pred_classes = torch.argmax(pred_scores, dim=-1)
                pred_confidences = torch.max(pred_scores, dim=-1)[0]
                
                # Filter by confidence
                valid_preds = pred_confidences > 0.3
                
                # Convert to numpy for visualization
                image_np = self._denormalize_image(original_image)
                image_pil = Image.fromarray(image_np)
                
                # Draw predictions
                draw = ImageDraw.Draw(image_pil)
                
                if valid_preds.sum() > 0:
                    valid_boxes = pred_boxes[valid_preds]
                    valid_classes = pred_classes[valid_preds]
                    valid_scores = pred_confidences[valid_preds]
                    
                    for box, cls, score in zip(valid_boxes, valid_classes, valid_scores):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        
                        # Ensure valid bounding box coordinates
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        cls_name = list(ROAD_CLASSES.keys())[cls.item()]
                        color = CLASS_COLORS[cls.item()]
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw label
                        label = f"{cls_name}: {score:.2f}"
                        draw.text((x1, y1-20), label, fill=color)
                
                # Draw ground truth
                for box, cls in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box.numpy()
                    cls_name = list(ROAD_CLASSES.keys())[cls.item()]
                    color = CLASS_COLORS[cls.item()]
                    
                    # Draw ground truth box (dashed)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
                    draw.text((x1, y1-40), f"GT: {cls_name}", fill=color)
                
                # Save visualization
                output_path = vis_dir / f"detection_sample_{i+1}.jpg"
                image_pil.save(output_path)
        
        print(f"[OK] Visualizations saved to: {vis_dir}")
    
    def _denormalize_image(self, image: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor for visualization."""
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        image = image.clone()
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        
        image = torch.clamp(image, 0, 1)
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        return image_np
    
    def plot_performance_metrics(self, results: Dict[str, Dict]):
        """
        Plot performance metrics across domains.
        
        Args:
            results: Dictionary of evaluation results
        """
        print("Generating performance plots...")
        
        # Extract data for plotting
        domains = list(results.keys())
        map_scores = [results[domain]['mAP'] for domain in domains]
        precision_scores = [results[domain]['precision'] for domain in domains]
        recall_scores = [results[domain]['recall'] for domain in domains]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ROAD Continual Learning Performance Metrics', fontsize=16)
        
        # mAP across domains
        axes[0, 0].bar(domains, map_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('mAP Across Domains')
        axes[0, 0].set_ylabel('mAP Score')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(map_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Precision across domains
        axes[0, 1].bar(domains, precision_scores, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Precision Across Domains')
        axes[0, 1].set_ylabel('Precision Score')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(precision_scores):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Recall across domains
        axes[1, 0].bar(domains, recall_scores, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Recall Across Domains')
        axes[1, 0].set_ylabel('Recall Score')
        axes[1, 0].set_ylim(0, 1)
        for i, v in enumerate(recall_scores):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Combined metrics
        x = np.arange(len(domains))
        width = 0.25
        
        axes[1, 1].bar(x - width, map_scores, width, label='mAP', alpha=0.7)
        axes[1, 1].bar(x, precision_scores, width, label='Precision', alpha=0.7)
        axes[1, 1].bar(x + width, recall_scores, width, label='Recall', alpha=0.7)
        
        axes[1, 1].set_title('All Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(domains)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'performance_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Performance plot saved to: {plot_path}")
    
    def compute_continual_learning_metrics(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute continual learning metrics (BWT, FWT).
        
        Args:
            results: Dictionary of evaluation results
            
        Returns:
            Dictionary of continual learning metrics
        """
        print("Computing continual learning metrics...")
        
        domains = list(results.keys())
        map_scores = [results[domain]['mAP'] for domain in domains]
        
        # Backward Transfer (BWT) - how much performance on early domains drops
        # Simplified calculation: compare first domain performance with final performance
        bwt = map_scores[0] - map_scores[-1] if len(map_scores) > 1 else 0.0
        
        # Forward Transfer (FWT) - how learning earlier domains helps on later ones
        # Simplified calculation: average improvement from first to last domain
        fwt = (map_scores[-1] - map_scores[0]) / len(map_scores) if len(map_scores) > 1 else 0.0
        
        # Average accuracy across all domains
        avg_accuracy = np.mean(map_scores)
        
        continual_metrics = {
            'backward_transfer': bwt,
            'forward_transfer': fwt,
            'average_accuracy': avg_accuracy,
            'domain_scores': dict(zip(domains, map_scores))
        }
        
        return continual_metrics
    
    def run_evaluation(self, checkpoint_name: str = "model_after_domain_4.pth"):
        """
        Run complete evaluation pipeline.
        
        Args:
            checkpoint_name: Name of the final model checkpoint
        """
        print("Starting evaluation pipeline...")
        
        # Load model
        model = self.load_model(checkpoint_name)
        
        # Load test datasets
        domain_datasets = self.avalanche_setup.load_domain_datasets()
        
        if not domain_datasets:
            raise ValueError("No domain datasets found!")
        
        # Evaluate on each domain
        evaluation_results = {}
        
        for domain in self.avalanche_setup.domains:
            if domain not in domain_datasets:
                print(f"Warning: Skipping domain {domain} - no data found")
                continue
            
            test_dataset = domain_datasets[domain]['test']
            
            # Compute mAP metrics
            metrics = self.compute_map_metrics(model, test_dataset, domain)
            evaluation_results[domain] = metrics
            
            # Generate visualizations
            self.visualize_detections(model, test_dataset, domain)
        
        # Save evaluation results
        results_path = self.output_dir / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate plots
        self.plot_performance_metrics(evaluation_results)
        
        # Compute continual learning metrics
        continual_metrics = self.compute_continual_learning_metrics(evaluation_results)
        
        # Save continual learning metrics
        continual_path = self.output_dir / 'continual_learning_metrics.json'
        with open(continual_path, 'w') as f:
            json.dump(continual_metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_dir}")
        print(f"mAP scores: {continual_metrics['domain_scores']}")
        print(f"Average accuracy: {continual_metrics['average_accuracy']:.3f}")
        print(f"Backward transfer: {continual_metrics['backward_transfer']:.3f}")
        print(f"Forward transfer: {continual_metrics['forward_transfer']:.3f}")

def main():
    """Main function to run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate RF-DETR model on ROAD dataset')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Directory containing trained models')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing test datasets')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, default='model_after_domain_4.pth',
                       help='Model checkpoint to evaluate')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ROADEvaluator(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.run_evaluation(checkpoint_name=args.checkpoint)

if __name__ == "__main__":
    main()
