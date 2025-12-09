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

# Run the create_outdoor_segmentation script
python create_outdoor_segmentation.py \
  --image_path wheat_inference_data/green_wheat_02.jpeg \
  --yolo_weights weights/gwc_yolo_weights.pt \
  --sam_checkpoint weights/sam_vit_h_4b8939.pth \
  --output_dir outputs/colour_transfer_testing/segmentations