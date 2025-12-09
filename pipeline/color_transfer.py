"""Color and texture transfer for infection application."""

import logging
from typing import Optional
import numpy as np
from PIL import Image
import cv2

from config.constants import (
    TARGET_LUMINANCE_MEAN,
    FEATHER_SIGMA,
    INPAINT_RADIUS
)
from utils.image_processing import (
    get_bbox_from_mask,
    edge_aware_mask_feathering,
    inpaint_background_gaussian
)


class ColorTransfer:
    """Color and texture transfer for applying infection to wheat heads."""
    
    def __init__(
        self,
        alpha: float = 1.0,
        darken: float = 0.3,
        contrast_strength: float = 0.7,
        color_temp_strength: float = 0.3,
        saturation_boost: float = 1.2,
        no_feathering: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize color transfer.
        
        Args:
            alpha: Transfer strength (0-1)
            darken: Darkening factor (0-1)
            contrast_strength: Local contrast preservation (0-1)
            color_temp_strength: Color temperature matching strength (0-1)
            saturation_boost: Saturation boost factor (1.0=no change)
            no_feathering: Disable mask feathering for sharper edges
            logger: Logger instance
        """
        self.alpha = alpha
        self.darken = darken
        self.contrast_strength = contrast_strength
        self.color_temp_strength = color_temp_strength
        self.saturation_boost = saturation_boost
        self.no_feathering = no_feathering
        self.logger = logger or logging.getLogger(__name__)
    
    def normalize_luminance(
        self,
        img_lab: np.ndarray,
        mask: np.ndarray,
        target_mean: float = TARGET_LUMINANCE_MEAN
    ) -> np.ndarray:
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
    
    def match_color_temperature(
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
    
    def preserve_local_contrast(
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
        source_bbox = get_bbox_from_mask(source_mask_np)
        target_bbox = get_bbox_from_mask(target_mask)
        
        if source_bbox is None or target_bbox is None:
            self.logger.warning("Cannot extract bounding box from mask")
            return target_image
        
        # Extract regions
        sy1, sy2, sx1, sx2 = source_bbox
        ty1, ty2, tx1, tx2 = target_bbox
        
        source_region = source_np[sy1:sy2, sx1:sx2]
        source_region_mask = source_mask_np[sy1:sy2, sx1:sx2]
        
        # Inpaint background for smooth transitions
        source_region = inpaint_background_gaussian(
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
        source_region_resized = self.match_color_temperature(
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
            mask_float = edge_aware_mask_feathering(
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
        source_lab_normalized = self.normalize_luminance(
            source_lab,
            source_mask_resized,
            target_mean=TARGET_LUMINANCE_MEAN
        )
        target_lab_normalized = self.normalize_luminance(
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
        result_lab = self.preserve_local_contrast(
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