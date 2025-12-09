"""Infected wheat head generation using SDXL ControlNet."""

import logging
from typing import Optional, Tuple
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel


class WheatGenerator:
    """Generate synthetic infected wheat heads using SDXL ControlNet."""
    
    def __init__(
        self,
        sdxl_weights_path: str,
        controlnet_weights_path: str,
        conditioning_image_path: str,
        device: str = "cuda",
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize wheat generator.
        
        Args:
            sdxl_weights_path: Path to SDXL base model
            controlnet_weights_path: Path to ControlNet weights
            conditioning_image_path: Path to conditioning image for ControlNet
            device: Device to run inference on
            logger: Logger instance
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
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
        
        # Load conditioning image
        self.conditioning_image = Image.open(conditioning_image_path).convert("RGB")
        self.logger.info(f"Loaded conditioning image from {conditioning_image_path}")
    
    def create_prompt_from_severity(self, severity: float) -> str:
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
    ) -> Image.Image:
        """
        Generate a synthetic infected wheat head using SDXL ControlNet.
        
        Args:
            infection_severity: Infection level (0-100%)
            seed: Random seed for reproducibility
        
        Returns:
            Generated infected wheat head image
        """
        self.logger.info(f"Generating infected wheat head at {infection_severity}% infection")
        
        # Create prompt
        prompt = self.create_prompt_from_severity(infection_severity)
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
        
        return output.images[0]