# Wheat Infection Pipeline

A synthetic Fusarium Head Blight (FHB) infection application pipeline for wheat head images. This project applies realistic infection patterns to outdoor wheat head images using deep learning techniques including YOLO detection, SAM segmentation, and SDXL ControlNet generation.

## Overview

This pipeline enables the generation of synthetic FHB-infected wheat head images from healthy outdoor wheat images. It supports two operational modes:

1. **Preselected Mode**: Transfer infection patterns from pre-made infected wheat head images
2. **Generation Mode**: Generate infected wheat heads using Stable Diffusion XL ControlNet with configurable infection severity

Both modes utilize:

- YOLO for wheat head detection in outdoor images
- Segment Anything Model (SAM) for precise segmentation
- Advanced color and texture transfer techniques

## Project Structure

```
.
├── main.py                 # Modular CLI entry point
├── condensed_main.py       # Standalone version (all code in one file, useful for HPC)
├── requirements.txt        # Python dependencies
├── config/                 # Configuration constants
├── pipeline/               # Core pipeline modules
│   ├── wheat_infection_pipeline.py
│   ├── detection.py        # YOLO + SAM detection
│   ├── generation.py       # SDXL ControlNet generation
│   └── color_transfer.py   # Color/texture transfer
├── utils/                  # Utility functions
├── scripts/                # Data preparation and training scripts
├── experiments/            # Testing and experimental scripts
├── outputs/                # Generated output images
├── wheat_training_data/    # Training dataset (single head images)
└── wheat_inference_data/   # Test images for inference
```

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- ~16GB GPU memory for SDXL mode

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Jared-Rost/fhb-image-translation.git
cd fhb-image-translation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required model weights:
   - **SAM checkpoint**: Download from [Segment Anything](https://github.com/facebookresearch/segment-anything)
   - **Indoor YOLO model**: Available at [FHB-Severity-Evaluation](https://github.com/RileyMccon/FHB-Severity-Evaluation)
   - **Outdoor YOLO model**: From [Global Wheat Challenge](https://github.com/ksnxr/GWC_solution)
   - **SDXL weights**: `stabilityai/stable-diffusion-xl-base-1.0` (auto-downloaded via HuggingFace)
   - **ControlNet weights**: Available upon request

## Usage

### Preselected Mode

Apply infection from a pre-made infected wheat head image:

```bash
python main.py preselected \
    --input_image outdoor.jpg \
    --source_infected_image infected.png \
    --source_mask mask.png \
    --sam_checkpoint models/sam_vit_h.pth \
    --yolo_outdoor models/outdoor_yolo.pt \
    --output_dir ./output
```

### Generation Mode

Generate synthetic infection with controllable severity:

```bash
python main.py generation \
    --input_image outdoor.jpg \
    --infection_severity 50.0 \
    --sam_checkpoint models/sam_vit_h.pth \
    --yolo_outdoor models/outdoor_yolo.pt \
    --yolo_indoor models/indoor_yolo.pt \
    --sdxl_weights stabilityai/stable-diffusion-xl-base-1.0 \
    --controlnet_weights ./controlnet_weights \
    --conditioning_image conditioning.png \
    --output_dir ./output \
    --seed 42
```

### Using Condensed Version

For HPC clusters or simplified deployment, use `condensed_main.py` (contains all code in a single file):

```bash
python condensed_main.py preselected [same arguments as above]
```

## Additional Scripts

### Data Preparation (`scripts/`)

- `create_metadata.py` - Generate training metadata from dataset
- `create_conditioning_images.py` - Create segmentation masks for ControlNet
- `resize_images.py` - Resize images to required dimensions
- `train_controlnet*.py` - Training scripts for ControlNet models

### Experiments (`experiments/`)

- `inference_test_*.py` - Testing scripts for model inference
- `visualize_outdoor_wheat_head_detection.py` - Visualize YOLO detections

## Data Sources & References

### Single Head Dataset

The training data for indoor wheat head images comes from McConachie et al. (2024):

- Paper: https://acsess.onlinelibrary.wiley.com/doi/10.1002/ppj2.20103
- GitHub: https://github.com/RileyMccon/FHB-Severity-Evaluation

### Models

**Segment Anything Model (SAM)**

- Source: https://github.com/facebookresearch/segment-anything

**Indoor YOLO Model**

- Source: McConachie et al. (2024) - https://github.com/RileyMccon/FHB-Severity-Evaluation

**Outdoor YOLO Model**

- Developed for the Global Wheat Challenge
- GitHub: https://github.com/ksnxr/GWC_solution
- Publication: https://doi.org/10.34133/2021/9846158

### Video Dataset

The outdoor wheat video frames used for testing are from Christopher Henry (private dataset).

## License

Copyright © 2025 Jared Rost

All rights reserved. This code is provided for review purposes only as part of an honours thesis project. Use, modification, or distribution of this code requires explicit written permission. For permission requests, please contact the repository owner.

## Acknowledgments

This project was completed as an honours thesis project. Special thanks to:

- Christopher Henry (supervisor) for providing video dataset and guidance
- McConachie et al. for the single head FHB dataset and indoor detection model
- The Global Wheat Challenge team for the outdoor detection model
- Meta AI for the Segment Anything Model
