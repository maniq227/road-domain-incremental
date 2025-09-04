#!/usr/bin/env python3
"""
Data Preparation Script for ROAD Continual Learning
==================================================

This script converts ROAD benchmark videos into COCO format image datasets
for continual learning with RF-DETR. It extracts frames from videos and
creates annotations for 11 active agent classes.

Author: ROAD Continual Learning Project
Date: 2024
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm
import random
from collections import defaultdict

# ROAD dataset class mappings for 11 active agent classes
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

# Weather domain mappings
WEATHER_DOMAINS = {
    'sunny': '2015-02-06-13-57-16_stereo_centre_01.mp4',
    'overcast': '2014-06-26-09-31-18_stereo_centre_02.mp4', 
    'snowy': '2015-02-03-08-45-10_stereo_centre_04.mp4',
    'night': '2014-12-10-18-10-50_stereo_centre_02.mp4'
}

class ROADDataProcessor:
    """Processes ROAD videos and converts them to COCO format datasets."""
    
    def __init__(self, video_dir: str, output_dir: str, frames_per_video: int = 100):
        """
        Initialize the data processor.
        
        Args:
            video_dir: Directory containing ROAD video files
            output_dir: Directory to save processed datasets
            frames_per_video: Number of frames to extract per video
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.frames_per_video = frames_per_video
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each domain
        for domain in WEATHER_DOMAINS.keys():
            (self.output_dir / domain / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / domain / 'annotations').mkdir(parents=True, exist_ok=True)
    
    def extract_frames_from_video(self, video_path: Path, domain: str) -> List[Dict]:
        """
        Extract frames from a video and create synthetic annotations.
        
        Args:
            video_path: Path to the video file
            domain: Weather domain name
            
        Returns:
            List of frame information dictionaries
        """
        print(f"Processing {domain} video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame indices to extract (uniformly distributed)
        frame_indices = np.linspace(0, total_frames-1, self.frames_per_video, dtype=int)
        
        frames_info = []
        frame_id = 0
        
        for idx in tqdm(frame_indices, desc=f"Extracting {domain} frames"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # Save frame
            frame_filename = f"{domain}_{frame_id:06d}.jpg"
            frame_path = self.output_dir / domain / 'images' / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create synthetic annotations for demonstration
            # In a real scenario, you would load actual ROAD annotations
            annotations = self._create_synthetic_annotations(frame, domain, frame_id)
            
            frame_info = {
                'id': frame_id,
                'file_name': frame_filename,
                'width': width,
                'height': height,
                'domain': domain,
                'annotations': annotations
            }
            
            frames_info.append(frame_info)
            frame_id += 1
        
        cap.release()
        return frames_info
    
    def _create_synthetic_annotations(self, frame: np.ndarray, domain: str, frame_id: int) -> List[Dict]:
        """
        Create synthetic annotations for demonstration purposes.
        In a real implementation, you would load actual ROAD annotations.
        
        Args:
            frame: Video frame
            domain: Weather domain
            frame_id: Frame identifier
            
        Returns:
            List of annotation dictionaries
        """
        height, width = frame.shape[:2]
        annotations = []
        
        # Generate 1-5 random objects per frame
        num_objects = random.randint(1, 5)
        
        for obj_id in range(num_objects):
            # Random class selection (biased towards common classes)
            class_weights = [0.4, 0.1, 0.1, 0.05, 0.05, 0.15, 0.05, 0.02, 0.02, 0.03, 0.03]
            class_id = np.random.choice(len(ROAD_CLASSES), p=class_weights)
            class_name = list(ROAD_CLASSES.keys())[class_id]
            
            # Generate random bounding box
            bbox_width = random.randint(50, min(200, width//4))
            bbox_height = random.randint(50, min(200, height//4))
            x = random.randint(0, width - bbox_width)
            y = random.randint(0, height - bbox_height)
            
            annotation = {
                'id': frame_id * 100 + obj_id,
                'image_id': frame_id,
                'category_id': class_id,
                'bbox': [x, y, bbox_width, bbox_height],
                'area': bbox_width * bbox_height,
                'iscrowd': 0
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def create_coco_dataset(self, frames_info: List[Dict], domain: str) -> Dict:
        """
        Create COCO format dataset from frame information.
        
        Args:
            frames_info: List of frame information dictionaries
            domain: Weather domain name
            
        Returns:
            COCO format dataset dictionary
        """
        # Create COCO format structure
        coco_dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories
        for class_name, class_id in ROAD_CLASSES.items():
            coco_dataset['categories'].append({
                'id': class_id,
                'name': class_name,
                'supercategory': 'vehicle' if 'vehicle' in class_name.lower() else 'agent'
            })
        
        # Add images and annotations
        for frame_info in frames_info:
            # Add image info
            image_info = {
                'id': frame_info['id'],
                'file_name': frame_info['file_name'],
                'width': frame_info['width'],
                'height': frame_info['height'],
                'domain': frame_info['domain']
            }
            coco_dataset['images'].append(image_info)
            
            # Add annotations
            for ann in frame_info['annotations']:
                coco_dataset['annotations'].append(ann)
        
        return coco_dataset
    
    def split_dataset(self, coco_dataset: Dict, train_ratio: float = 0.8) -> Tuple[Dict, Dict]:
        """
        Split dataset into train and test sets.
        
        Args:
            coco_dataset: COCO format dataset
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        images = coco_dataset['images']
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        # Create train dataset
        train_dataset = {
            'images': train_images,
            'annotations': [],
            'categories': coco_dataset['categories']
        }
        
        # Create test dataset
        test_dataset = {
            'images': test_images,
            'annotations': [],
            'categories': coco_dataset['categories']
        }
        
        # Assign annotations to train/test based on image IDs
        train_image_ids = {img['id'] for img in train_images}
        test_image_ids = {img['id'] for img in test_images}
        
        for ann in coco_dataset['annotations']:
            if ann['image_id'] in train_image_ids:
                train_dataset['annotations'].append(ann)
            elif ann['image_id'] in test_image_ids:
                test_dataset['annotations'].append(ann)
        
        return train_dataset, test_dataset
    
    def process_all_domains(self):
        """Process all weather domains and create COCO datasets."""
        print("Starting ROAD data preparation...")
        print(f"Extracting {self.frames_per_video} frames per video")
        
        all_domains_info = {}
        
        for domain, video_filename in WEATHER_DOMAINS.items():
            video_path = self.video_dir / video_filename
            
            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                continue
            
            # Extract frames
            frames_info = self.extract_frames_from_video(video_path, domain)
            
            # Create COCO dataset
            coco_dataset = self.create_coco_dataset(frames_info, domain)
            
            # Split into train/test
            train_dataset, test_dataset = self.split_dataset(coco_dataset)
            
            # Save datasets
            train_path = self.output_dir / domain / 'annotations' / 'train.json'
            test_path = self.output_dir / domain / 'annotations' / 'test.json'
            
            with open(train_path, 'w') as f:
                json.dump(train_dataset, f, indent=2)
            
            with open(test_path, 'w') as f:
                json.dump(test_dataset, f, indent=2)
            
            # Store domain info
            all_domains_info[domain] = {
                'train_images': len(train_dataset['images']),
                'test_images': len(test_dataset['images']),
                'train_annotations': len(train_dataset['annotations']),
                'test_annotations': len(test_dataset['annotations'])
            }
            
            print(f"[OK] {domain}: {len(train_dataset['images'])} train, {len(test_dataset['images'])} test images")
        
        # Save summary
        summary_path = self.output_dir / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(all_domains_info, f, indent=2)
        
        print(f"\nDataset preparation complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"Summary saved to: {summary_path}")
        
        return all_domains_info

def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(description='Prepare ROAD dataset for continual learning')
    parser.add_argument('--video_dir', type=str, default='videos',
                       help='Directory containing ROAD video files')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for processed datasets')
    parser.add_argument('--frames_per_video', type=int, default=100,
                       help='Number of frames to extract per video')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Process data
    processor = ROADDataProcessor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        frames_per_video=args.frames_per_video
    )
    
    domain_info = processor.process_all_domains()
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for domain, info in domain_info.items():
        print(f"{domain.upper():>10}: {info['train_images']:>3} train, {info['test_images']:>3} test images")
        print(f"{'':>10}  {info['train_annotations']:>3} train, {info['test_annotations']:>3} test annotations")

if __name__ == "__main__":
    main()
