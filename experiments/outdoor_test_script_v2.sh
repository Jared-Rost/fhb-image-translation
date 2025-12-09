#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source $HOME/env/wheat_infection_filter_env/bin/activate

# Paths
OUTDOOR_DIR="wheat_inference_data/2023_video_data"
INDOOR_DIR="wheat_training_data/preselected_1024"
MASK_DIR="wheat_training_data/preselected_1024_segmentation_masks"
OUTPUT_BASE="outputs/2023_video_testing"
SAM_CHECKPOINT="weights/sam_vit_h_4b8939.pth"
YOLO_OUTDOOR="weights/gwc_yolo_weights.pt"

# Create base output directory
mkdir -p "$OUTPUT_BASE"

# Enable nullglob to handle cases where no files match
shopt -s nullglob

# Loop through all indoor wheat head images
for indoor_image in "$INDOOR_DIR"/*.jpg "$INDOOR_DIR"/*.jpeg "$INDOOR_DIR"/*.png "$INDOOR_DIR"/*.JPG "$INDOOR_DIR"/*.JPEG "$INDOOR_DIR"/*.PNG; do
    # Get the base filename without extension
    indoor_basename=$(basename "$indoor_image")
    indoor_name="${indoor_basename%.*}"
    
    # Construct the corresponding mask path
    indoor_mask="${MASK_DIR}/${indoor_name}_segmentation.png"
    
    # Check if mask exists
    if [ ! -f "$indoor_mask" ]; then
        echo "Warning: Mask not found for $indoor_basename, skipping..."
        continue
    fi
    
    # Create output directory for this indoor image
    output_dir="${OUTPUT_BASE}/${indoor_name}"
    mkdir -p "$output_dir"
    
    echo "=========================================="
    echo "Processing with indoor image: $indoor_basename"
    echo "Using mask: $(basename $indoor_mask)"
    echo "Output directory: $output_dir"
    echo "=========================================="
    
    # Loop through all outdoor images
    for outdoor_image in "$OUTDOOR_DIR"/*.jpg "$OUTDOOR_DIR"/*.jpeg "$OUTDOOR_DIR"/*.png "$OUTDOOR_DIR"/*.JPG "$OUTDOOR_DIR"/*.JPEG "$OUTDOOR_DIR"/*.PNG; do
        echo "  -> Processing outdoor image: $(basename $outdoor_image)"
        
        python main.py preselected \
            --input_image "$outdoor_image" \
            --source_infected_image "$indoor_image" \
            --source_mask "$indoor_mask" \
            --sam_checkpoint "$SAM_CHECKPOINT" \
            --yolo_outdoor "$YOLO_OUTDOOR" \
            --output_dir "$output_dir" \
            --alpha 1.0 \
            --contrast_strength 0.7 \
            --darken 0.0 \
            --color_temp_strength 0.0 \
            --saturation_boost 1.0 \
            --device cuda \
            --log_level INFO
    done
    
    echo "Completed processing for $indoor_basename"
    echo ""
done

echo "=========================================="
echo "All processing complete!"
echo "Results saved to $OUTPUT_BASE"
echo "=========================================="