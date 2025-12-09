"""Wheat head detection using YOLO and SAM."""

import logging
from typing import List, Dict, Optional
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import cv2

from config.constants import SAM_MODEL_TYPE


class WheatDetector:
    """Wheat head detection using YOLO + SAM segmentation."""
    
    def __init__(
        self,
        sam_checkpoint_path: str,
        yolo_outdoor_path: str,
        yolo_indoor_path: Optional[str] = None,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize wheat detector.
        
        Args:
            sam_checkpoint_path: Path to SAM checkpoint
            yolo_outdoor_path: Path to YOLO weights for outdoor multi-head detection
            yolo_indoor_path: Path to YOLO weights for indoor single head detection
            device: Device to run inference on
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # Load SAM
        self.logger.info(f"Loading SAM from {sam_checkpoint_path}")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_checkpoint_path)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Load outdoor YOLO (YOLOv5)
        self.logger.info(f"Loading YOLOv5 outdoor detector from {yolo_outdoor_path}")
        self.yolo_outdoor = torch.hub.load(
            'ultralytics/yolov5:v7.0',
            'custom',
            path=yolo_outdoor_path,
            trust_repo=True
        )
        if device == "cuda":
            self.yolo_outdoor.to(device)
        
        # Load indoor YOLO (YOLOv8) if provided
        self.yolo_indoor = None
        if yolo_indoor_path:
            self.logger.info(f"Loading YOLOv8 indoor detector from {yolo_indoor_path}")
            self.yolo_indoor = YOLO(yolo_indoor_path)
    
    def detect_wheat_heads_in_scene(self, image_path: str) -> List[Dict]:
        """
        Detect all wheat heads in an outdoor scene image.
        
        Args:
            image_path: Path to outdoor wheat head image
        
        Returns:
            List of detection results with bounding boxes and masks
        """
        self.logger.info(f"Detecting wheat heads in {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect wheat heads with YOLOv5
        results = self.yolo_outdoor(image_rgb)
        
        # Convert YOLOv5 results to bounding boxes
        boxes_df = results.pandas().xyxy[0]
        
        detections = []
        self.sam_predictor.set_image(image_rgb)

        if len(boxes_df) == 0:
            self.logger.warning("No wheat heads detected in scene")
            return []
        
        for idx, row in boxes_df.iterrows():
            bbox = np.array([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence = row['confidence']
            
            # Get segmentation mask with SAM
            masks, _, _ = self.sam_predictor.predict(
                box=bbox,
                multimask_output=False
            )
            
            detections.append({
                'bbox': bbox,
                'confidence': confidence,
                'mask': masks[0]
            })
        
        self.logger.info(f"Detected {len(detections)} wheat heads")
        return detections
    
    def segment_single_head(
        self,
        image: Image.Image,
        use_indoor_yolo: bool = True
    ) -> Image.Image:
        """
        Segment a single wheat head using YOLO + SAM.
        
        Args:
            image: Input image containing a single wheat head
            use_indoor_yolo: If True, use indoor YOLO model, otherwise use outdoor
        
        Returns:
            Binary segmentation mask
        """
        self.logger.debug("Segmenting single wheat head")
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Detect wheat head with YOLO
        yolo_model = self.yolo_indoor if use_indoor_yolo else self.yolo_outdoor
        
        if use_indoor_yolo and self.yolo_indoor is None:
            raise RuntimeError("Indoor YOLO model not loaded")
        
        results = yolo_model(image_np)
        
        # Handle different YOLO versions
        if use_indoor_yolo:
            # YOLOv8 format
            if len(results[0].boxes) == 0:
                self.logger.warning("No wheat head detected in image")
                return Image.new("L", image.size, color=0)
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
        else:
            # YOLOv5 format
            boxes = results.pandas().xyxy[0][["xmin", "ymin", "xmax", "ymax"]].values
            if len(boxes) == 0:
                self.logger.warning("No wheat head detected in image")
                return Image.new("L", image.size, color=0)
            box = boxes[0]
        
        # Use SAM to get precise segmentation
        self.sam_predictor.set_image(image_np)
        masks, _, _ = self.sam_predictor.predict(
            box=box,
            multimask_output=False
        )
        
        # Convert mask to PIL Image
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        
        return mask_image