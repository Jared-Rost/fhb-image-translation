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
OUTPUT_DIR="outputs/detection_visualizations"
SAM_CHECKPOINT="weights/sam_vit_h_4b8939.pth"
YOLO_OUTDOOR="weights/gwc_yolo_weights.pt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Enable nullglob to handle cases where no files match
shopt -s nullglob

echo "=========================================="
echo "Visualizing wheat head detections"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Process all images in the input directory
for input_image in "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png "$INPUT_DIR"/*.JPG "$INPUT_DIR"/*.JPEG "$INPUT_DIR"/*.PNG; do
    # Get the base filename
    input_basename=$(basename "$input_image")
    input_name="${input_basename%.*}"
    
    # Create output filename
    output_image="${OUTPUT_DIR}/${input_name}_detected.png"
    
    echo "Processing: $input_basename"
    
    python visualize_outdoor_wheat_head_detection.py \
        --image "$input_image" \
        --sam_checkpoint "$SAM_CHECKPOINT" \
        --yolo_weights "$YOLO_OUTDOOR" \
        --output "$output_image" \
        --device cuda
    
    echo ""
done

echo "=========================================="
echo "All visualizations complete!"
echo "Results saved to $OUTPUT_DIR"
echo "=========================================="