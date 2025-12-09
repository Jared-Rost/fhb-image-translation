#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=01-00:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source ~/env/wheat_infection_filter_env/bin/activate

# Run the conditioning image creation script
python create_conditioning_images.py \
  --input_dir wheat_training_data/database_single_head_images_resized_1024 \
  --output_dir wheat_training_data/segmentation_masks_resized_1024 \
  --yolo_weights weights/wheat_detection_yolo_weights.pt \
  --sam_checkpoint weights/sam_vit_h_4b8939.pth \
  --log_level INFO