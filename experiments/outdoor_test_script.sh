#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source $HOME/env/wheat_infection_filter_env/bin/activate

# Paths
INPUT_DIR="wheat_inference_data/2023_video_data"
OUTPUT_DIR="outputs/2023_video_testing"
SOURCE_IMAGE="wheat_training_data/preselected_1024/8_Ava_1_2_14DAI.png"
SOURCE_MASK="wheat_training_data/preselected_1024_segmentation_masks/8_Ava_1_2_14DAI_segmentation.png"
SAM_CHECKPOINT="weights/sam_vit_h_4b8939.pth"
YOLO_OUTDOOR="weights/gwc_yolo_weights.pt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process all images in the input directory
shopt -s nullglob  # Prevent literal glob patterns when no matches found
for input_image in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png "$INPUT_DIR"/*.JPG "$INPUT_DIR"/*.JPEG "$INPUT_DIR"/*.PNG; do
    echo "Processing: $input_image"
    
    python main.py preselected \
        --input_image "$input_image" \
        --source_infected_image "$SOURCE_IMAGE" \
        --source_mask "$SOURCE_MASK" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --yolo_outdoor "$YOLO_OUTDOOR" \
        --output_dir "$OUTPUT_DIR" \
        --alpha 1.0 \
        --contrast_strength 0.7 \
        --darken 0.0 \
        --color_temp_strength 0.0 \
        --saturation_boost 1.0 \
        --device cuda \
        --log_level INFO
done

echo "Processing complete! Results saved to $OUTPUT_DIR"