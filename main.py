#!/usr/bin/env python3
"""
Wheat Infection Pipeline - CLI Entry Point

Applies synthetic infection to outdoor wheat head images using two modes:
1. Preselected Mode: Use pre-made infected wheat head image + segmentation mask
2. Generation Mode: Generate infected wheat head using SDXL ControlNet with infection score

Both modes use YOLO + SAM for outdoor wheat head detection and color/texture transfer.
"""

import argparse
from pipeline import WheatInfectionPipeline


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
        default=0.3,
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
        default=0.3,
        help='Color temperature matching strength (0-1)'
    )
    preselected_parser.add_argument(
        '--saturation_boost',
        type=float,
        default=1.2,
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