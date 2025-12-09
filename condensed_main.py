#!/usr/bin/env python3
"""
Wheat Infection Pipeline

Applies synthetic infection to outdoor wheat head images using two modes:
1. Preselected Mode: Use pre-made infected wheat head image + segmentation mask
2. Generation Mode: Generate infected wheat head using SDXL ControlNet with infection score

Both modes use YOLO + SAM for outdoor wheat head detection and color/texture transfer.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from ultralytics import YOLO  # For YOLOv8 (indoor detector)
from segment_anything import sam_model_registry, SamPredictor
import cv2


# Constants for color transfer
BBOX_PADDING = 5
INPAINT_RADIUS = 5
FEATHER_SIGMA = 2
TARGET_LUMINANCE_MEAN = 128


class WheatInfectionPipeline:
    """Pipeline for applying synthetic infection to wheat head images."""
    
    def __init__(
        self,
        sam_checkpoint_path: str,
        yolo_outdoor_path: str,
        device: str = "cuda",
        log_level: str = "INFO",
        # Generation mode parameters (optional)
        sdxl_weights_path: Optional[str] = None,
        controlnet_weights_path: Optional[str] = None,
        yolo_indoor_path: Optional[str] = None,
        conditioning_image_path: Optional[str] = None,
        # Color transfer parameters
        alpha: float = 1.0,
        darken: float = 0.0,
        contrast_strength: float = 0.7,
        color_temp_strength: float = 0.0,
        saturation_boost: float = 1.0,
        no_feathering: bool = False,
    ):
        """
        Initialize the wheat infection pipeline.
        
        Args:
            sam_checkpoint_path: Path to SAM checkpoint
            yolo_outdoor_path: Path to YOLO weights for outdoor multi-head detection
            device: Device to run inference on ("cuda" or "cpu")
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            sdxl_weights_path: Path to SDXL base model (for generation mode)
            controlnet_weights_path: Path to ControlNet weights (for generation mode)
            yolo_indoor_path: Path to YOLO weights for indoor single head detection (for generation mode)
            conditioning_image_path: Path to conditioning image for ControlNet (for generation mode)
            alpha: Transfer strength (0-1)
            darken: Darkening factor (0-1)
            contrast_strength: Local contrast preservation (0-1)
            color_temp_strength: Color temperature matching strength (0-1)
            saturation_boost: Saturation boost factor (1.0=no change)
            no_feathering: Disable mask feathering for sharper edges
        """
        self.device = device
        self.logger = self._setup_logger(log_level)
        
        # Store color transfer parameters
        self.alpha = alpha
        self.darken = darken
        self.contrast_strength = contrast_strength
        self.color_temp_strength = color_temp_strength
        self.saturation_boost = saturation_boost
        self.no_feathering = no_feathering
        
        self.logger.info("Initializing Wheat Infection Pipeline...")
        
        # Load SAM (required for both modes)
        self.logger.info(f"Loading SAM from {sam_checkpoint_path}")
        sam_model_type = "vit_h"  # Adjust based on your checkpoint
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
        # Load outdoor YOLO (YOLOv5) - required for both modes
        self.logger.info(f"Loading YOLOv5 outdoor detector from {yolo_outdoor_path}")
        self.yolo_outdoor = torch.hub.load('ultralytics/yolov5:v7.0', 'custom', path=yolo_outdoor_path, trust_repo=True)
        if device == "cuda":
            self.yolo_outdoor.to(device)
        
        # Load generation mode models if provided
        self.generation_mode_enabled = False
        if all([sdxl_weights_path, controlnet_weights_path, yolo_indoor_path, conditioning_image_path]):
            self.logger.info("Initializing generation mode components...")
            
            # Load SDXL + ControlNet
            self.logger.info(f"Loading SDXL from {sdxl_weights_path}")
            self.logger.info(f"Loading ControlNet from {controlnet_weights_path}")
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_weights_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            self.sdxl_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                sdxl_weights_path,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None
            )
            self.sdxl_pipeline = self.sdxl_pipeline.to(device)
            
            # Load indoor YOLO (YOLOv8)
            self.logger.info(f"Loading YOLOv8 indoor detector from {yolo_indoor_path}")
            self.yolo_indoor = YOLO(yolo_indoor_path)
            
            # Load conditioning image
            self.conditioning_image = Image.open(conditioning_image_path).convert("RGB")
            self.logger.info(f"Loaded conditioning image from {conditioning_image_path}")
            
            self.generation_mode_enabled = True
            self.logger.info("Generation mode enabled")
        else:
            self.logger.info("Generation mode not enabled (missing parameters)")
        
        self.logger.info("Pipeline initialization complete!")
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("WheatInfectionPipeline")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _create_prompt_from_severity(self, severity: float) -> str:
        """
        Create SDXL prompt from infection severity score.
        
        Args:
            severity: Infection severity (0-100%)
        
        Returns:
            Formatted prompt string
        """
        if severity == 0:
            return "AINBN3tpDb, no Fusarium Head Blight, 0.0%"
        elif 0 < severity <= 10:
            return f"AINBN3tpDb, minimal Fusarium Head Blight, {severity:.1f}%"
        elif 10 < severity <= 30:
            return f"AINBN3tpDb, mild Fusarium Head Blight, {severity:.1f}%"
        elif 30 < severity <= 60:
            return f"AINBN3tpDb, moderate Fusarium Head Blight, {severity:.1f}%"
        elif 60 < severity <= 100:
            return f"AINBN3tpDb, severe Fusarium Head Blight, {severity:.1f}%"
        else:
            return f"AINBN3tpDb, unknown Fusarium Head Blight, {severity:.1f}%"
    
    def generate_infected_wheat_head(
        self,
        infection_severity: float,
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Generate a synthetic infected wheat head using SDXL ControlNet.
        
        Args:
            infection_severity: Infection level (0-100%)
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (generated_image, segmentation_mask)
        """
        if not self.generation_mode_enabled:
            raise RuntimeError("Generation mode not enabled. Please provide all required parameters during initialization.")
        
        self.logger.info(f"Generating infected wheat head at {infection_severity}% infection")
        
        # Create prompt
        prompt = self._create_prompt_from_severity(infection_severity)
        self.logger.debug(f"Using prompt: {prompt}")
        
        # Generate image
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)
        
        output = self.sdxl_pipeline(
            prompt=prompt,
            image=self.conditioning_image,
            num_inference_steps=50,
            generator=generator,
            guidance_scale=7.5
        )
        
        generated_image = output.images[0]
        
        # Generate segmentation mask using indoor YOLO + SAM
        segmentation_mask = self._segment_single_head(generated_image, use_indoor_yolo=True)
        
        return generated_image, segmentation_mask
    
    def _segment_single_head(
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
    
    def _get_bbox_from_mask(self, mask: np.ndarray, padding: int = BBOX_PADDING) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate bounding box coordinates from a binary mask.
        
        Args:
            mask: Binary mask (boolean array)
            padding: Number of pixels to pad around the bounding box
            
        Returns:
            Tuple of (y1, y2, x1, x2) or None if mask is empty
        """
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return None
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        y1 = max(0, y1 - padding)
        x1 = max(0, x1 - padding)
        y2 = min(mask.shape[0], y2 + padding)
        x2 = min(mask.shape[1], x2 + padding)
        return (y1, y2, x1, x2)

    def _edge_aware_mask_feathering(self, mask: np.ndarray, image: np.ndarray, sigma: float = FEATHER_SIGMA) -> np.ndarray:
        """
        Feather mask while respecting edge structure.
        Uses bilateral filter to preserve edges.
        
        Args:
            mask: Binary mask
            image: Reference image for edge detection
            sigma: Spatial sigma for bilateral filter
            
        Returns:
            Feathered mask as float array (0-1)
        """
        mask_float = mask.astype(np.float32)
        mask_uint8 = (mask_float * 255).astype(np.uint8)
        
        feathered = cv2.bilateralFilter(
            mask_uint8,
            d=9,
            sigmaColor=75,
            sigmaSpace=sigma
        )
        
        return feathered.astype(np.float32) / 255.0

    def _normalize_luminance(self, img_lab: np.ndarray, mask: np.ndarray, target_mean: float = TARGET_LUMINANCE_MEAN) -> np.ndarray:
        """
        Normalize luminance to a fixed mean, preserving relative variation.
        
        Args:
            img_lab: Image in LAB color space
            mask: Binary mask of region to normalize
            target_mean: Target mean luminance value
            
        Returns:
            Normalized LAB image
        """
        img_normalized = img_lab.copy()
        
        if mask.sum() > 0:
            current_mean = img_lab[:, :, 0][mask].mean()
            img_normalized[:, :, 0] = img_lab[:, :, 0] - current_mean + target_mean
        
        return img_normalized

    def _match_color_temperature(
        self,
        src_rgb: np.ndarray,
        tgt_rgb: np.ndarray,
        src_mask: np.ndarray,
        tgt_mask: np.ndarray,
        strength: float = 0.3
    ) -> np.ndarray:
        """
        Adjust source image's color temperature to match target.
        
        Args:
            src_rgb: Source RGB image
            tgt_rgb: Target RGB image
            src_mask: Source mask
            tgt_mask: Target mask
            strength: Color temperature matching strength (0-1)
            
        Returns:
            Color-adjusted source image
        """
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            return src_rgb
        
        src_avg = src_rgb[src_mask].mean(axis=0)
        tgt_avg = tgt_rgb[tgt_mask].mean(axis=0)
        
        color_shift = tgt_avg - src_avg
        
        src_adjusted = src_rgb.astype(np.float32)
        src_adjusted += color_shift * strength
        src_adjusted = np.clip(src_adjusted, 0, 255).astype(np.uint8)
        
        return src_adjusted

    def _preserve_local_contrast(
        self,
        result_lab: np.ndarray,
        original_lab: np.ndarray,
        mask: np.ndarray,
        strength: float = 0.1
    ) -> np.ndarray:
        """
        Preserve original image's local contrast in L channel.
        
        Args:
            result_lab: Result image in LAB space
            original_lab: Original image in LAB space
            mask: Binary mask of region to process
            strength: Strength of contrast preservation (0-1)
            
        Returns:
            LAB image with preserved contrast
        """
        original_l = original_lab[:, :, 0].astype(np.float32)
        local_mean_original = cv2.GaussianBlur(original_l, (7, 7), 0)
        local_variance_original = cv2.GaussianBlur((original_l - local_mean_original)**2, (7, 7), 0)
        local_std_original = np.sqrt(np.maximum(local_variance_original, 1e-6))
        
        result_l = result_lab[:, :, 0].astype(np.float32)
        local_mean_result = cv2.GaussianBlur(result_l, (7, 7), 0)
        local_variance_result = cv2.GaussianBlur((result_l - local_mean_result)**2, (7, 7), 0)
        local_std_result = np.sqrt(np.maximum(local_variance_result, 1e-6))
        
        contrast_ratio = local_std_original / np.maximum(local_std_result, 1e-6)
        contrast_adjusted = local_mean_result + (result_l - local_mean_result) * contrast_ratio * strength
        
        result_lab[:, :, 0] = result_lab[:, :, 0].astype(np.float32)
        result_lab[:, :, 0][mask] = contrast_adjusted[mask]
        
        return result_lab

    def _inpaint_background_gaussian(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        inpaint_radius: int = INPAINT_RADIUS
    ) -> np.ndarray:
        """
        Fill background areas using Gaussian inpainting based on nearby wheat colors.
        Creates smooth color transitions from wheat edges into background.
        
        Args:
            img: RGB image
            mask: Boolean mask of wheat region
            inpaint_radius: Radius for inpainting algorithm
        
        Returns:
            Image with background inpainted using colors from wheat edges
        """
        if mask.sum() == 0:
            return img
        
        # Convert mask to uint8 format required by cv2.inpaint
        # 0 = areas to inpaint (background), 255 = known areas (wheat)
        inpaint_mask = (~mask).astype(np.uint8) * 255
        
        # Use Telea inpainting algorithm (fast marching method)
        result = cv2.inpaint(img, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        return result
    
    def transfer_infection_texture(
        self,
        source_image: Image.Image,
        source_mask: Image.Image,
        target_image: np.ndarray,
        target_mask: np.ndarray
    ) -> np.ndarray:
        """
        Transfer infected wheat appearance from source to target wheat head.
        Uses improved color transfer techniques with LAB color space, lighting preservation,
        and local contrast preservation.
        
        Args:
            source_image: Generated or preselected infected wheat head
            source_mask: Segmentation mask of source
            target_image: Real outdoor wheat image
            target_mask: Mask of wheat head to modify
    
        Returns:
            Modified target image with transferred infection
        """
        self.logger.debug("Transferring infection texture and color")
        
        # Convert PIL to numpy
        source_np = np.array(source_image)
        source_mask_np = np.array(source_mask) > 127
        
        # Get bounding boxes
        source_bbox = self._get_bbox_from_mask(source_mask_np)
        target_bbox = self._get_bbox_from_mask(target_mask)
        
        if source_bbox is None or target_bbox is None:
            self.logger.warning("Cannot extract bounding box from mask")
            return target_image
        
        # Extract regions
        sy1, sy2, sx1, sx2 = source_bbox
        ty1, ty2, tx1, tx2 = target_bbox
        
        source_region = source_np[sy1:sy2, sx1:sx2]
        source_region_mask = source_mask_np[sy1:sy2, sx1:sx2]
        
        # Inpaint background for smooth transitions
        source_region = self._inpaint_background_gaussian(
            source_region,
            source_region_mask,
            inpaint_radius=INPAINT_RADIUS
        )
        
        target_region = target_image[ty1:ty2, tx1:tx2]
        target_region_mask = target_mask[ty1:ty2, tx1:tx2]
        
        # Resize source to match target size
        target_h, target_w = target_region.shape[:2]
        source_region_resized = cv2.resize(source_region, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        source_mask_resized = cv2.resize(
            source_region_mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        # Use target mask only - ensures complete coverage
        combined_mask = target_region_mask
        
        if combined_mask.sum() == 0:
            self.logger.warning("Empty mask, skipping")
            return target_image
        
        # Color temperature matching
        source_region_resized = self._match_color_temperature(
            source_region_resized,
            target_region,
            source_mask_resized,
            target_region_mask,
            strength=self.color_temp_strength
        )
        
        # Edge-aware mask feathering (optional)
        if self.no_feathering:
            mask_float = combined_mask.astype(np.float32)
        else:
            mask_float = self._edge_aware_mask_feathering(
                combined_mask,
                target_region,
                sigma=FEATHER_SIGMA
            )
        
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_RGB2LAB).astype(np.float32)
        source_lab = cv2.cvtColor(source_region_resized, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Save target's original luminance
        target_luminance_original = target_lab[:, :, 0].copy()
        
        # Normalize both to same luminance level
        source_lab_normalized = self._normalize_luminance(
            source_lab,
            source_mask_resized,
            target_mean=TARGET_LUMINANCE_MEAN
        )
        target_lab_normalized = self._normalize_luminance(
            target_lab,
            target_region_mask,
            target_mean=TARGET_LUMINANCE_MEAN
        )
        
        # Color transfer (a, b channels) with alpha blending
        result_lab = target_lab_normalized.copy()
        
        for c in [1, 2]:  # Transfer a* and b* channels
            result_lab[:, :, c] = (
                self.alpha * source_lab_normalized[:, :, c] * mask_float +
                (1 - self.alpha * mask_float) * target_lab_normalized[:, :, c]
            )
        
        # Apply darkening for infected appearance
        if combined_mask.sum() > 0:
            src_l_mean = source_lab[:, :, 0][source_mask_resized].mean()
            tgt_l_mean = target_lab[:, :, 0][target_region_mask].mean()
            
            # Match source darkness
            luminance_adjustment = (src_l_mean - tgt_l_mean) * self.darken
            
            self.logger.debug(f"Luminance adjustment: {luminance_adjustment:.1f}")
            
            # Apply darkening
            result_lab[:, :, 0] = target_luminance_original.copy()
            result_lab[:, :, 0][combined_mask] = np.clip(
                target_luminance_original[combined_mask] + luminance_adjustment,
                0, 255
            )
        
        # Preserve local contrast
        result_lab = self._preserve_local_contrast(
            result_lab,
            target_lab,
            combined_mask,
            strength=self.contrast_strength
        )
        
        # Boost saturation
        if self.saturation_boost != 1.0 and combined_mask.sum() > 0:
            result_lab[:, :, 1][combined_mask] = np.clip(
                128 + (result_lab[:, :, 1][combined_mask] - 128) * self.saturation_boost,
                0, 255
            )
            result_lab[:, :, 2][combined_mask] = np.clip(
                128 + (result_lab[:, :, 2][combined_mask] - 128) * self.saturation_boost,
                0, 255
            )
        
        # Convert back to RGB
        result_lab = np.clip(result_lab, 0, 255)
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # Put result back into full image
        out_img = target_image.copy()
        out_img[ty1:ty2, tx1:tx2] = result
        
        return out_img

    def process_image_preselected_mode(
        self,
        input_image_path: str,
        source_infected_image_path: str,
        source_mask_path: str,
        output_dir: str
    ) -> str:
        """
        Preselected mode: Apply pre-made infected wheat head to outdoor scene.
        
        Args:
            input_image_path: Path to outdoor wheat head image
            source_infected_image_path: Path to pre-made infected wheat head image
            source_mask_path: Path to segmentation mask of infected wheat head
            output_dir: Directory to save output images
        
        Returns:
            Path to output image
        """
        self.logger.info(f"Processing {input_image_path} in PRESELECTED mode")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load source infected wheat head and mask
        self.logger.info(f"Loading source infected wheat head from {source_infected_image_path}")
        source_image = Image.open(source_infected_image_path).convert("RGB")
        
        self.logger.info(f"Loading source mask from {source_mask_path}")
        source_mask = Image.open(source_mask_path).convert("L")
        
        # Detect wheat heads in outdoor scene
        self.logger.info("Detecting wheat heads in outdoor scene")
        detections = self.detect_wheat_heads_in_scene(input_image_path)
        
        if len(detections) == 0:
            self.logger.warning("No wheat heads detected in outdoor image")
            return None
        
        # Load original image
        outdoor_image = cv2.imread(input_image_path)
        outdoor_image_rgb = cv2.cvtColor(outdoor_image, cv2.COLOR_BGR2RGB)
        result_image = outdoor_image_rgb.copy()
        
        # Apply infection to each detected wheat head
        self.logger.info(f"Applying infection to {len(detections)} wheat heads")
        for i, detection in enumerate(detections):
            self.logger.debug(f"Processing wheat head {i+1}/{len(detections)}")
            
            result_image = self.transfer_infection_texture(
                source_image,
                source_mask,
                result_image,
                detection['mask']
            )
        
        # Generate output filename (same as input)
        input_filename = Path(input_image_path).name
        output_path = os.path.join(output_dir, input_filename)
        
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_bgr)
        
        self.logger.info(f"Saved result to {output_path}")
        return output_path

    def process_image_generation_mode(
        self,
        input_image_path: str,
        infection_severity: float,
        output_dir: str,
        seed: Optional[int] = None
    ) -> str:
        """
        Generation mode: Generate infected wheat head and apply to outdoor scene.
        
        Args:
            input_image_path: Path to outdoor wheat head image
            infection_severity: Infection level (0-100%)
            output_dir: Directory to save output images
            seed: Random seed for reproducibility
        
        Returns:
            Path to output image
        """
        self.logger.info(f"Processing {input_image_path} in GENERATION mode with {infection_severity}% infection")
        
        if not self.generation_mode_enabled:
            raise RuntimeError("Generation mode not enabled. Please provide all required parameters during initialization.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate infected wheat head
        self.logger.info("Generating synthetic infected wheat head")
        source_image, source_mask = self.generate_infected_wheat_head(
            infection_severity, seed
        )
        
        # Save intermediate results for debugging
        source_image.save(os.path.join(output_dir, "generated_infected_wheat.png"))
        source_mask.save(os.path.join(output_dir, "generated_mask.png"))
        
        # Detect wheat heads in outdoor scene
        self.logger.info("Detecting wheat heads in outdoor scene")
        detections = self.detect_wheat_heads_in_scene(input_image_path)
        
        if len(detections) == 0:
            self.logger.warning("No wheat heads detected in outdoor image")
            return None
        
        # Load original image
        outdoor_image = cv2.imread(input_image_path)
        outdoor_image_rgb = cv2.cvtColor(outdoor_image, cv2.COLOR_BGR2RGB)
        result_image = outdoor_image_rgb.copy()
        
        # Apply infection to each detected wheat head
        self.logger.info(f"Applying infection to {len(detections)} wheat heads")
        for i, detection in enumerate(detections):
            self.logger.debug(f"Processing wheat head {i+1}/{len(detections)}")
            
            result_image = self.transfer_infection_texture(
                source_image,
                source_mask,
                result_image,
                detection['mask']
            )
        
        # Generate output filename (same as input)
        input_filename = Path(input_image_path).name
        output_path = os.path.join(output_dir, input_filename)
        
        result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result_image_bgr)
        
        self.logger.info(f"Saved result to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Wheat Infection Pipeline - Apply synthetic FHB infection to outdoor wheat images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Preselected mode:
    python main.py preselected \\
        --input_image outdoor.jpg \\
        --source_infected_image infected.png \\
        --source_mask mask.png \\
        --sam_checkpoint sam_vit_h.pth \\
        --yolo_outdoor outdoor_yolo.pt \\
        --output_dir ./output
  
  Generation mode:
    python main.py generation \\
        --input_image outdoor.jpg \\
        --infection_severity 50.0 \\
        --sam_checkpoint sam_vit_h.pth \\
        --yolo_outdoor outdoor_yolo.pt \\
        --yolo_indoor indoor_yolo.pt \\
        --sdxl_weights stabilityai/stable-diffusion-xl-base-1.0 \\
        --controlnet_weights ./controlnet_weights \\
        --conditioning_image conditioning.png \\
        --output_dir ./output \\
        --seed 42
        """
    )
    
    # Add subparsers for modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode', required=True)
    
    # Preselected mode parser
    preselected_parser = subparsers.add_parser(
        'preselected',
        help='Use pre-made infected wheat head image'
    )
    preselected_parser.add_argument(
        '--input_image',
        type=str,
        required=True,
        help='Path to outdoor wheat head image'
    )
    preselected_parser.add_argument(
        '--source_infected_image',
        type=str,
        required=True,
        help='Path to pre-made infected wheat head image'
    )
    preselected_parser.add_argument(
        '--source_mask',
        type=str,
        required=True,
        help='Path to segmentation mask of infected wheat head (PNG)'
    )
    preselected_parser.add_argument(
        '--sam_checkpoint',
        type=str,
        required=True,
        help='Path to SAM checkpoint'
    )
    preselected_parser.add_argument(
        '--yolo_outdoor',
        type=str,
        required=True,
        help='Path to YOLOv5 weights for outdoor multi-head detection'
    )
    preselected_parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output images'
    )
    preselected_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    preselected_parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    # Color transfer parameters for preselected mode
    preselected_parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Transfer strength (0-1)'
    )
    preselected_parser.add_argument(
        '--darken',
        type=float,
        default=0.0,
        help='Darkening factor (0-1)'
    )
    preselected_parser.add_argument(
        '--contrast_strength',
        type=float,
        default=0.7,
        help='Local contrast preservation (0-1)'
    )
    preselected_parser.add_argument(
        '--color_temp_strength',
        type=float,
        default=0.0,
        help='Color temperature matching strength (0-1)'
    )
    preselected_parser.add_argument(
        '--saturation_boost',
        type=float,
        default=1.0,
        help='Saturation boost factor (1.0=no change)'
    )
    preselected_parser.add_argument(
        '--no_feathering',
        action='store_true',
        help='Disable mask feathering for sharper edges'
    )
    
    # Generation mode parser
    generation_parser = subparsers.add_parser(
        'generation',
        help='Generate infected wheat head using SDXL ControlNet'
    )
    generation_parser.add_argument(
        '--input_image',
        type=str,
        required=True,
        help='Path to outdoor wheat head image'
    )
    generation_parser.add_argument(
        '--infection_severity',
        type=float,
        required=True,
        help='Infection severity score (0-100%%)'
    )
    generation_parser.add_argument(
        '--sam_checkpoint',
        type=str,
        required=True,
        help='Path to SAM checkpoint'
    )
    generation_parser.add_argument(
        '--yolo_outdoor',
        type=str,
        required=True,
        help='Path to YOLOv5 weights for outdoor multi-head detection'
    )
    generation_parser.add_argument(
        '--yolo_indoor',
        type=str,
        required=True,
        help='Path to YOLOv8 weights for indoor single-head detection'
    )
    generation_parser.add_argument(
        '--sdxl_weights',
        type=str,
        required=True,
        help='Path to SDXL base model (e.g., stabilityai/stable-diffusion-xl-base-1.0)'
    )
    generation_parser.add_argument(
        '--controlnet_weights',
        type=str,
        required=True,
        help='Path to fine-tuned ControlNet weights'
    )
    generation_parser.add_argument(
        '--conditioning_image',
        type=str,
        required=True,
        help='Path to conditioning image for ControlNet'
    )
    generation_parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output images'
    )
    generation_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    generation_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on'
    )
    generation_parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    # Color transfer parameters for generation mode
    generation_parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Transfer strength (0-1)'
    )
    generation_parser.add_argument(
        '--darken',
        type=float,
        default=0.0,
        help='Darkening factor (0-1)'
    )
    generation_parser.add_argument(
        '--contrast_strength',
        type=float,
        default=0.7,
        help='Local contrast preservation (0-1)'
    )
    generation_parser.add_argument(
        '--color_temp_strength',
        type=float,
        default=0.0,
        help='Color temperature matching strength (0-1)'
    )
    generation_parser.add_argument(
        '--saturation_boost',
        type=float,
        default=1.0,
        help='Saturation boost factor (1.0=no change)'
    )
    generation_parser.add_argument(
        '--no_feathering',
        action='store_true',
        help='Disable mask feathering for sharper edges'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline based on mode
    if args.mode == 'preselected':
        pipeline = WheatInfectionPipeline(
            sam_checkpoint_path=args.sam_checkpoint,
            yolo_outdoor_path=args.yolo_outdoor,
            device=args.device,
            log_level=args.log_level,
            alpha=args.alpha,
            darken=args.darken,
            contrast_strength=args.contrast_strength,
            color_temp_strength=args.color_temp_strength,
            saturation_boost=args.saturation_boost,
            no_feathering=args.no_feathering
        )
        
        pipeline.process_image_preselected_mode(
            input_image_path=args.input_image,
            source_infected_image_path=args.source_infected_image,
            source_mask_path=args.source_mask,
            output_dir=args.output_dir
        )
    
    elif args.mode == 'generation':
        pipeline = WheatInfectionPipeline(
            sam_checkpoint_path=args.sam_checkpoint,
            yolo_outdoor_path=args.yolo_outdoor,
            device=args.device,
            log_level=args.log_level,
            sdxl_weights_path=args.sdxl_weights,
            controlnet_weights_path=args.controlnet_weights,
            yolo_indoor_path=args.yolo_indoor,
            conditioning_image_path=args.conditioning_image,
            alpha=args.alpha,
            darken=args.darken,
            contrast_strength=args.contrast_strength,
            color_temp_strength=args.color_temp_strength,
            saturation_boost=args.saturation_boost,
            no_feathering=args.no_feathering
        )
        
        pipeline.process_image_generation_mode(
            input_image_path=args.input_image,
            infection_severity=args.infection_severity,
            output_dir=args.output_dir,
            seed=args.seed
        )


if __name__ == "__main__":
    main()