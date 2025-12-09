#!/usr/bin/env python3
"""
Semantic Segmentation Script for Conditioning Image Creation

This script processes all images in a directory through a machine learning pipeline.

Usage:
    python create-conditioning-images.py --input_dir /path/to/images --output_dir /path/to/output
"""

import os
import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('segmentation.log'),
            logging.StreamHandler()
        ]
    )


class ProcessingPipeline:
    """Pipeline for processing images."""
    
    def __init__(self, yolo_weights_path: str, sam_checkpoint_path: str):
        """Initialize the processing pipeline.
        
        Args:
            yolo_weights_path: Path to YOLO model weights file
            sam_checkpoint_path: Path to SAM checkpoint file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing processing pipeline...")
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model from {yolo_weights_path}")
        self.yolo_model = YOLO(yolo_weights_path)
        self.logger.info("YOLO model loaded successfully")
        
        # Load SAM model
        self.logger.info(f"Loading SAM model from {sam_checkpoint_path}")
        model_type = "vit_h"  # Default to vit_h as in the SAM accuracy notebook
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self.logger.info("SAM model loaded successfully")
        
    def process_single_image(self, image_path: str, output_dir: str) -> bool:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save outputs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_name = Path(image_path).stem
            self.logger.info(f"Processing {image_name}...")
            
            # Run YOLO model on the image
            results = self.yolo_model.predict(image_path, conf=0.25)

            # If still no detections, try even lower
            if not any(len(result.boxes) > 0 for result in results):
                self.logger.info(f"Trying lower confidence for {image_name}")
                results = self.yolo_model.predict(image_path, conf=0.05)  # Very low threshold
            
            # Extract bounding box coordinates from the detection
            bboxes = []
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        bbox = box.xyxy.tolist()[0]  # Get [xmin, ymin, xmax, ymax]
                        bboxes.append(bbox)
                        self.logger.debug(f"Found wheat head at bbox: {bbox}")
            
            if not bboxes:
                self.logger.warning(f"No wheat heads detected in {image_name}")
                return True  # Consider this successful but with no detections
            
            self.logger.info(f"Detected {len(bboxes)} wheat heads in {image_name}")
            
            # Load image for SAM processing
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image)
            
            # Create a combined segmentation mask for all detected wheat heads
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Process each bounding box with SAM
            for i, bbox in enumerate(bboxes):
                self.logger.debug(f"Processing bbox {i+1}/{len(bboxes)}: {bbox}")
                
                # Convert bbox to numpy array as expected by SAM
                input_box = np.array(bbox)
                
                # Get segmentation mask from SAM
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Add this mask to the combined mask
                # Convert boolean mask to uint8 and add to combined mask
                mask_uint8 = (masks[0] * 255).astype(np.uint8)
                combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)
            
            # Save the segmentation mask
            output_filename = f"{image_name}_segmentation.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save the combined mask
            cv2.imwrite(output_path, combined_mask)
            self.logger.info(f"Saved segmentation mask: {output_filename}")
            
            self.logger.info(f"Successfully processed {image_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from input directory.
    
    Args:
        input_dir: Input directory path
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        # Also check uppercase extensions
        pattern = os.path.join(input_dir, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    
    return sorted(image_files)


def main():
    """Main function to run the processing pipeline."""
    parser = argparse.ArgumentParser(description='Image Processing Pipeline')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output images')
    parser.add_argument('--yolo_weights', type=str, required=True,
                        help='Path to YOLO model weights file')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Get image files
    image_files = get_image_files(args.input_dir)
    if not image_files:
        logger.error(f"No image files found in {args.input_dir}")
        return 1
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    try:
        # Initialize pipeline
        pipeline = ProcessingPipeline(args.yolo_weights, args.sam_checkpoint)
        
        # Process images
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            if pipeline.process_single_image(image_path, args.output_dir):
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info(f"Processing complete! Successfully processed: {successful}, Failed: {failed}")
        
        if failed > 0:
            return 1
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
