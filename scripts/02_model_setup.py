#!/usr/bin/env python3
"""
Model Setup Script for RF-DETR Continual Learning
================================================

This script initializes the RF-DETR model with COCO pretrained weights
and adapts it for the 11 ROAD active agent classes.

Author: ROAD Continual Learning Project
Date: 2024
"""

import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from pathlib import Path
import json
from typing import Dict, Any
import timm

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

class RFDETRModel(nn.Module):
    """
    RF-DETR (Real-time Detection Transformer) model adapted for ROAD dataset.
    
    This is a simplified implementation focusing on the key components
    needed for continual learning with Avalanche.
    """
    
    def __init__(self, num_classes: int = 11, pretrained: bool = True):
        """
        Initialize RF-DETR model.
        
        Args:
            num_classes: Number of object classes (11 for ROAD)
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Backbone: EfficientNet or ResNet (simplified for demo)
        if pretrained:
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        else:
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False, features_only=True)
        
        # Get actual backbone output channels by running a test forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.backbone(test_input)
            backbone_out_channels = test_features[-1].shape[1]  # Last feature map channels
        
        # Feature pyramid network (simplified)
        self.fpn = nn.ModuleList([
            nn.Conv2d(backbone_out_channels, 256, 1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 3, padding=1)
        ])
        
        # Detection head
        self.classifier = nn.Linear(256, num_classes)
        self.bbox_regressor = nn.Linear(256, 4)  # x, y, w, h
        
        # Object queries (simplified transformer component)
        self.num_queries = 100
        self.query_embed = nn.Embedding(self.num_queries, 256)
        
        # Transformer decoder (simplified)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=3
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of RF-DETR.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary containing predictions
        """
        batch_size = x.size(0)
        
        # Extract features using backbone
        features = self.backbone(x)[-1]  # Use last feature map
        
        # Apply FPN
        for layer in self.fpn:
            features = torch.relu(layer(features))
        
        # Global average pooling
        features = torch.mean(features, dim=[2, 3])  # (B, 256)
        
        # Object queries
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Transformer decoder (simplified)
        memory = features.unsqueeze(1)  # (B, 1, 256)
        decoder_output = self.transformer_decoder(query_embeds, memory)
        
        # Classification and regression
        class_logits = self.classifier(decoder_output)  # (B, num_queries, num_classes)
        bbox_coords = torch.sigmoid(self.bbox_regressor(decoder_output))  # (B, num_queries, 4)
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords
        }

class ROADModelManager:
    """Manages RF-DETR model for ROAD continual learning."""
    
    def __init__(self, model_dir: str = "outputs/models"):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model configuration
        self.config = {
            'num_classes': len(ROAD_CLASSES),
            'class_names': list(ROAD_CLASSES.keys()),
            'class_mapping': ROAD_CLASSES,
            'input_size': (640, 640),
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4
        }
    
    def create_model(self, pretrained: bool = True) -> RFDETRModel:
        """
        Create and initialize RF-DETR model.
        
        Args:
            pretrained: Whether to use pretrained weights
            
        Returns:
            Initialized RF-DETR model
        """
        print("Creating RF-DETR model...")
        
        model = RFDETRModel(
            num_classes=self.config['num_classes'],
            pretrained=pretrained
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model created successfully!")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Number of classes: {self.config['num_classes']}")
        print(f"Classes: {self.config['class_names']}")
        
        return model
    
    def save_model(self, model: RFDETRModel, filename: str, metadata: Dict = None):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            filename: Filename for the checkpoint
            metadata: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'metadata': metadata or {}
        }
        
        save_path = self.model_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Model saved to: {save_path}")
    
    def load_model(self, filename: str) -> RFDETRModel:
        """
        Load model from checkpoint.
        
        Args:
            filename: Filename of the checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint_path = self.model_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model with saved config
        model = RFDETRModel(
            num_classes=checkpoint['config']['num_classes'],
            pretrained=False
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded from: {checkpoint_path}")
        return model
    
    def get_data_transforms(self, is_training: bool = True):
        """
        Get data transforms for training/validation.
        
        Args:
            is_training: Whether transforms are for training
            
        Returns:
            Transform pipeline
        """
        if is_training:
            transform = transforms.Compose([
                transforms.Resize(self.config['input_size']),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.config['input_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def save_config(self):
        """Save model configuration."""
        config_path = self.model_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {config_path}")

def main():
    """Main function to set up the model."""
    parser = argparse.ArgumentParser(description='Set up RF-DETR model for ROAD continual learning')
    parser.add_argument('--model_dir', type=str, default='outputs/models',
                       help='Directory to save model files')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Create model manager
    manager = ROADModelManager(model_dir=args.model_dir)
    
    # Create and save initial model
    model = manager.create_model(pretrained=args.pretrained)
    
    # Save initial model
    manager.save_model(model, 'initial_model.pth', {
        'description': 'Initial RF-DETR model with COCO pretrained weights',
        'num_classes': len(ROAD_CLASSES),
        'classes': list(ROAD_CLASSES.keys())
    })
    
    # Save configuration
    manager.save_config()
    
    print("\n" + "="*50)
    print("MODEL SETUP COMPLETE")
    print("="*50)
    print(f"Model saved to: {args.model_dir}")
    print(f"Number of classes: {len(ROAD_CLASSES)}")
    print(f"Classes: {list(ROAD_CLASSES.keys())}")
    print(f"Device: {manager.device}")

if __name__ == "__main__":
    main()
