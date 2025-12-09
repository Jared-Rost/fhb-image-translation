"""Constants and default parameters for wheat infection pipeline."""

# Image processing constants
BBOX_PADDING = 5
INPAINT_RADIUS = 5
FEATHER_SIGMA = 2
TARGET_LUMINANCE_MEAN = 128

# Default color transfer parameters
DEFAULT_ALPHA = 1.0
DEFAULT_DARKEN = 0.3
DEFAULT_CONTRAST_STRENGTH = 0.7
DEFAULT_COLOR_TEMP_STRENGTH = 0.3
DEFAULT_SATURATION_BOOST = 1.2

# Model types
SAM_MODEL_TYPE = "vit_h"