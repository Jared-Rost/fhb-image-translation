"""Image processing utilities for wheat infection pipeline."""

from typing import Optional, Tuple
import numpy as np
import cv2

from config.constants import BBOX_PADDING, FEATHER_SIGMA, INPAINT_RADIUS


def get_bbox_from_mask(
    mask: np.ndarray,
    padding: int = BBOX_PADDING
) -> Optional[Tuple[int, int, int, int]]:
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


def edge_aware_mask_feathering(
    mask: np.ndarray,
    image: np.ndarray,
    sigma: float = FEATHER_SIGMA
) -> np.ndarray:
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


def inpaint_background_gaussian(
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