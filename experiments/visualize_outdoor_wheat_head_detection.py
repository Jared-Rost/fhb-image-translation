#!/usr/bin/env python3
"""
Visualize Outdoor Wheat Head Detection

Uses YOLOv5 and SAM to detect wheat heads in outdoor images and saves a visualization
with all detected wheat heads highlighted with colored masks.
"""

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
import cv2


def detect_and_visualize_wheat_heads(
    image_path: str,
    sam_checkpoint: str,
    yolo_weights: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Detect wheat heads using YOLOv5 and SAM, then save a visualization.
    
    Args:
        image_path: Path to outdoor wheat head image
        sam_checkpoint: Path to SAM checkpoint
        yolo_weights: Path to YOLOv5 weights
        output_path: Path to save the visualization
        device: Device to run inference on ("cuda" or "cpu")
    """
    print(f"Loading models...")
    
    # Load SAM
    sam_model_type = "vit_h"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    # Load YOLOv5
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights)
    if device == "cuda":
        yolo_model.to(device)
    
    print(f"Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect wheat heads with YOLOv5
    results = yolo_model(image_rgb)
    boxes_df = results.pandas().xyxy[0]
    
    print(f"Detected {len(boxes_df)} wheat heads with YOLO")
    
    # Get SAM masks for each detection
    masks = []
    confidences = []
    sam_predictor.set_image(image_rgb)
    
    for idx, row in boxes_df.iterrows():
        bbox = np.array([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        confidence = row['confidence']
        
        # Get segmentation mask with SAM
        mask_output, _, _ = sam_predictor.predict(
            box=bbox,
            multimask_output=False
        )
        
        masks.append(mask_output[0])
        confidences.append(confidence)
    
    print(f"Generated {len(masks)} segmentation masks")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    
    # Create a combined mask with different colors for each wheat head
    if len(masks) > 0:
        combined_mask = np.zeros((*masks[0].shape, 3), dtype=np.float32)
        
        # Generate distinct colors for each mask
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            # Create colored mask
            mask_bool = mask > 0
            for c in range(3):
                combined_mask[:, :, c] += mask_bool * colors[i, c]
        
        # Normalize combined mask
        max_val = combined_mask.max()
        if max_val > 0:
            combined_mask = combined_mask / max_val
        
        plt.imshow(combined_mask, alpha=0.5)
    
    plt.title(f"Detected {len(masks)} Wheat Heads")
    plt.axis("off")
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()
    
    # Print detection summary
    print("\nDetection Summary:")
    for i, conf in enumerate(confidences):
        print(f"  Wheat Head {i+1}: Confidence = {conf:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and visualize outdoor wheat heads using YOLOv5 + SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python visualize_wheat_detection.py \\
        --image wheat_inference_data/2023_video_data/2023_video_01.png \\
        --sam_checkpoint weights/sam_vit_h_4b8939.pth \\
        --yolo_weights weights/gwc_yolo_weights.pt \\
        --output outputs/detection_viz/2023_video_01_detected.png
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to outdoor wheat head image'
    )
    parser.add_argument(
        '--sam_checkpoint',
        type=str,
        required=True,
        help='Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)'
    )
    parser.add_argument(
        '--yolo_weights',
        type=str,
        required=True,
        help='Path to YOLOv5 weights for outdoor detection'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save the visualization image'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Run detection and visualization
    detect_and_visualize_wheat_heads(
        image_path=args.image,
        sam_checkpoint=args.sam_checkpoint,
        yolo_weights=args.yolo_weights,
        output_path=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()