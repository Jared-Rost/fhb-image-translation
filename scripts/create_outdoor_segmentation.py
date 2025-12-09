#!/usr/bin/env python3
"""
Generate wheat head detections and segmentation masks for an outdoor image using YOLOv5 and SAM.

Usage:
    python create_outdoor_segmentation.py \
        --image_path path/to/outdoor_image.jpg \
        --yolo_weights path/to/yolov5_multi_head.pt \
        --sam_checkpoint path/to/sam_checkpoint.pth \
        --output_dir path/to/output_dir
"""

import argparse
import numpy as np
from pathlib import Path
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

def main():
    parser = argparse.ArgumentParser(description="Generate wheat head detections and masks using YOLOv5 and SAM")
    parser.add_argument("--image_path", type=str, required=True, help="Path to outdoor image")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to YOLOv5 weights for multi head detection")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detections (.npz)")
    parser.add_argument("--sam_model_type", type=str, default="vit_h", help="SAM model type (default: vit_h)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Prepare output path
    input_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (input_path.stem + "_detections.npz")

    # Load image
    image = cv2.imread(str(args.image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load YOLOv5 model using torch.hub (requirements version)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.yolo_weights)
    model.to(args.device)
    results = model(image_rgb)

    # Load SAM
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)
    sam_predictor.set_image(image_rgb)

    # Get detections and masks
    masks_list = []
    bboxes_list = []
    confidences_list = []

    # YOLOv5 results: results.xyxy[0] columns: x1, y1, x2, y2, conf, class
    for box in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, conf, cls = box
        bbox = np.array([x1, y1, x2, y2])
        confidence = conf
        masks, _, _ = sam_predictor.predict(
            box=bbox,
            multimask_output=False
        )
        mask = masks[0]  # shape (H, W), bool
        masks_list.append(mask.astype(np.uint8))
        bboxes_list.append(bbox)
        confidences_list.append(confidence)

    # Save detections to .npz
    np.savez_compressed(
        output_file,
        masks=np.stack(masks_list, axis=0) if masks_list else np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8),
        bboxes=np.stack(bboxes_list, axis=0) if bboxes_list else np.zeros((0, 4)),
        confidences=np.array(confidences_list) if confidences_list else np.zeros((0,))
    )
    print(f"Saved {len(masks_list)} detections to {output_file}")

if __name__ == "__main__":
    main()