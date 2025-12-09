#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source ~/env/wheat_infection_filter_env/bin/activate

# Run the main pipeline in generation mode
python main.py generation \
  --input_image wheat_inference_data/2023_video_data/2023_video_01.png \
  --infection_severity 50.0 \
  --sam_checkpoint weights/sam_vit_h_4b8939.pth \
  --yolo_outdoor weights/wheat_detection_yolo_weights.pt \
  --yolo_indoor weights/gwc_yolo_weights.pt \
  --sdxl_weights stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_weights controlnet_training_output/v1_sdxl \
  --conditioning_image wheat_training_data/segmentation_masks_resized_1024/8_Ava_1_2_14DAI_segmentation.png \
  --output_dir outputs/final_test \
  --seed 42 \
  --device cuda \
  --log_level INFO