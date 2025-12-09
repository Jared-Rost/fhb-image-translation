"""Main wheat infection pipeline orchestrator."""

import os
from pathlib import Path
from typing import Optional
import cv2
from PIL import Image

from utils.logging_config import setup_logger
from .detection import WheatDetector
from .generation import WheatGenerator
from .color_transfer import ColorTransfer


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
        darken: float = 0.3,
        contrast_strength: float = 0.7,
        color_temp_strength: float = 0.3,
        saturation_boost: float = 1.2,
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
        self.logger = setup_logger("WheatInfectionPipeline", log_level)
        
        self.logger.info("Initializing Wheat Infection Pipeline...")
        
        # Initialize detector (required for both modes)
        self.detector = WheatDetector(
            sam_checkpoint_path=sam_checkpoint_path,
            yolo_outdoor_path=yolo_outdoor_path,
            yolo_indoor_path=yolo_indoor_path,
            device=device,
            logger=self.logger
        )
        
        # Initialize color transfer
        self.color_transfer = ColorTransfer(
            alpha=alpha,
            darken=darken,
            contrast_strength=contrast_strength,
            color_temp_strength=color_temp_strength,
            saturation_boost=saturation_boost,
            no_feathering=no_feathering,
            logger=self.logger
        )
        
        # Initialize generator if generation mode parameters provided
        self.generator = None
        self.generation_mode_enabled = False
        
        if all([sdxl_weights_path, controlnet_weights_path, conditioning_image_path]):
            self.logger.info("Initializing generation mode components...")
            self.generator = WheatGenerator(
                sdxl_weights_path=sdxl_weights_path,
                controlnet_weights_path=controlnet_weights_path,
                conditioning_image_path=conditioning_image_path,
                device=device,
                logger=self.logger
            )
            self.generation_mode_enabled = True
            self.logger.info("Generation mode enabled")
        else:
            self.logger.info("Generation mode not enabled (missing parameters)")
        
        self.logger.info("Pipeline initialization complete!")
    
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
        detections = self.detector.detect_wheat_heads_in_scene(input_image_path)
        
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
            
            result_image = self.color_transfer.transfer_infection_texture(
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
        source_image = self.generator.generate_infected_wheat_head(
            infection_severity, seed
        )
        
        # Segment generated wheat head
        source_mask = self.detector.segment_single_head(source_image, use_indoor_yolo=True)
        
        # Save intermediate results for debugging
        source_image.save(os.path.join(output_dir, "generated_infected_wheat.png"))
        source_mask.save(os.path.join(output_dir, "generated_mask.png"))
        
        # Detect wheat heads in outdoor scene
        self.logger.info("Detecting wheat heads in outdoor scene")
        detections = self.detector.detect_wheat_heads_in_scene(input_image_path)
        
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
            
            result_image = self.color_transfer.transfer_infection_texture(
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