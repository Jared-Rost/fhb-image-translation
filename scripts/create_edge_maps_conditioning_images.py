#!/usr/bin/env python3
"""
Edge Map Generation Script for SDXL ControlNet Conditioning

Creates Canny edge maps from wheat head segmentation masks.

Usage:
    python create_edge_maps_conditioning_images.py \
        --input_dir /path/to/images \
        --output_dir /path/to/edge_maps \
        --yolo_weights /path/to/yolo/weights \
        --sam_checkpoint /path/to/sam/checkpoint
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
            logging.FileHandler('edge_map_generation.log'),
            logging.StreamHandler()
        ]
    )


def create_edge_map_from_mask(mask: np.ndarray, 
                               low_threshold: int = 50, 
                               high_threshold: int = 150,
                               blur_kernel: int = 5) -> np.ndarray:
    """
    Create Canny edge map from segmentation mask.
    
    Args:
        mask: Binary segmentation mask (0 or 255)
        low_threshold: Canny lower threshold
        high_threshold: Canny upper threshold
        blur_kernel: Gaussian blur kernel size (odd number)
        
    Returns:
        Edge map (0 or 255)
    """
    # Ensure mask is binary
    mask_binary = (mask > 127).astype(np.uint8) * 255
    
    # Optional: Slight blur to smooth edges before Canny
    if blur_kernel > 1:
        mask_binary = cv2.GaussianBlur(mask_binary, (blur_kernel, blur_kernel), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(mask_binary, low_threshold, high_threshold)
    
    return edges


class EdgeMapPipeline:
    """Pipeline for creating edge maps from images."""
    
    def __init__(self, yolo_weights_path: str, sam_checkpoint_path: str,
                 canny_low: int = 50, canny_high: int = 150):
        """Initialize the edge map generation pipeline.
        
        Args:
            yolo_weights_path: Path to YOLO model weights file
            sam_checkpoint_path: Path to SAM checkpoint file
            canny_low: Canny lower threshold
            canny_high: Canny upper threshold
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing edge map pipeline...")
        
        # Canny parameters
        self.canny_low = canny_low
        self.canny_high = canny_high
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.logger.info(f"Loading YOLO model from {yolo_weights_path}")
        self.yolo_model = YOLO(yolo_weights_path)
        self.logger.info("YOLO model loaded successfully")
        
        # Load SAM model
        self.logger.info(f"Loading SAM model from {sam_checkpoint_path}")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
        self.logger.info("SAM model loaded successfully")
        
    def process_single_image(self, image_path: str, output_dir: str, 
                            save_intermediate: bool = False) -> bool:
        """
        Process a single image to create edge map.
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save outputs
            save_intermediate: If True, also save segmentation masks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_name = Path(image_path).stem
            self.logger.info(f"Processing {image_name}...")
            
            # Run YOLO model on the image
            results = self.yolo_model.predict(image_path, conf=0.25)

            # If no detections, try lower confidence
            if not any(len(result.boxes) > 0 for result in results):
                self.logger.info(f"Trying lower confidence for {image_name}")
                results = self.yolo_model.predict(image_path, conf=0.05)
            
            # Extract bounding box coordinates
            bboxes = []
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    for box in boxes:
                        bbox = box.xyxy.tolist()[0]
                        bboxes.append(bbox)
                        self.logger.debug(f"Found wheat head at bbox: {bbox}")
            
            if not bboxes:
                self.logger.warning(f"No wheat heads detected in {image_name}")
                return True
            
            self.logger.info(f"Detected {len(bboxes)} wheat heads in {image_name}")
            
            # Load image for SAM processing
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image)
            
            # Create combined segmentation mask
            h, w = image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Process each bounding box with SAM
            for i, bbox in enumerate(bboxes):
                self.logger.debug(f"Processing bbox {i+1}/{len(bboxes)}: {bbox}")
                
                input_box = np.array(bbox)
                
                # Get segmentation mask from SAM
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Add to combined mask
                mask_uint8 = (masks[0] * 255).astype(np.uint8)
                combined_mask = cv2.bitwise_or(combined_mask, mask_uint8)
            
            # Save intermediate segmentation mask if requested
            if save_intermediate:
                seg_filename = f"{image_name}_segmentation.png"
                seg_path = os.path.join(output_dir, seg_filename)
                cv2.imwrite(seg_path, combined_mask)
                self.logger.info(f"Saved segmentation mask: {seg_filename}")
            
            # Create edge map from combined mask
            edge_map = create_edge_map_from_mask(
                combined_mask, 
                low_threshold=self.canny_low,
                high_threshold=self.canny_high
            )
            
            # Save edge map
            edge_filename = f"{image_name}_edges.png"
            edge_path = os.path.join(output_dir, edge_filename)
            cv2.imwrite(edge_path, edge_map)
            self.logger.info(f"Saved edge map: {edge_filename}")
            
            self.logger.info(f"Successfully processed {image_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {e}")
            return False


def get_image_files(input_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all image files from input directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    
    return sorted(image_files)


def main():
    """Main function to run the edge map generation pipeline."""
    parser = argparse.ArgumentParser(description='Edge Map Generation for SDXL ControlNet')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save edge maps')
    parser.add_argument('--yolo_weights', type=str, required=True,
                        help='Path to YOLO model weights file')
    parser.add_argument('--sam_checkpoint', type=str, required=True,
                        help='Path to SAM checkpoint file')
    parser.add_argument('--canny_low', type=int, default=50,
                        help='Canny lower threshold (default: 50)')
    parser.add_argument('--canny_high', type=int, default=150,
                        help='Canny upper threshold (default: 150)')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate segmentation masks')
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
        pipeline = EdgeMapPipeline(
            args.yolo_weights, 
            args.sam_checkpoint,
            canny_low=args.canny_low,
            canny_high=args.canny_high
        )
        
        # Process images
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
            
            if pipeline.process_single_image(image_path, args.output_dir, 
                                            save_intermediate=args.save_intermediate):
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
