#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=01-00:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source $HOME/env/wheat_infection_filter_env/bin/activate

# Directory containing infected wheat head images
INFECTED_DIR="wheat_training_data/database_single_head_images_resized_1024"
# Directory containing segmentation masks for infected images
SEGMENTATION_DIR="wheat_training_data/segmentation_masks_resized_1024"
# Healthy reference image (source)
HEALTHY_IMAGE="wheat_training_data/database_single_head_images_resized_1024/180376_Ava_1_3_7DAI.png"
# Healthy reference mask
HEALTHY_MASK="wheat_training_data/segmentation_masks_resized_1024/180376_Ava_1_3_7DAI_segmentation.png"
# Output directory for artificial healthy wheat heads
OUTPUT_DIR="outputs/colour_transfer_testing/artificial_healthy_wheat_heads"

# Loop through all infected wheat head images
for img in ${INFECTED_DIR}/*.png; do
    base=$(basename "$img" .png)
    infected_mask="${SEGMENTATION_DIR}/${base}_segmentation.png"
    
    # Run color transfer: healthy -> infected to create artificial healthy
    python colour_transfer_v2_segmentation_map_healthy.py \
        --infected_image "$img" \
        --infected_mask "$infected_mask" \
        --healthy_image "$HEALTHY_IMAGE" \
        --healthy_mask "$HEALTHY_MASK" \
        --output_dir "$OUTPUT_DIR" \
        --alpha 1.0 \
        --brighten 0.2 \
        --contrast_strength 0.7
done