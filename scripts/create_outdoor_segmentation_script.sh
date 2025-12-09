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

# Array of image names to process
IMAGE_NAMES=(
  green_wheat_03.webp
  green_wheat_04.webp
  green_wheat_05.webp
  green_wheat_06.webp
  green_wheat_07.jpg
  green_wheat_08.jpg
  green_wheat_09.jpeg
  green_wheat_10.jpg
  green_wheat_11.jpg
  # Add more image names as needed
)

for IMAGE_NAME in "${IMAGE_NAMES[@]}"; do
  echo "Processing $IMAGE_NAME"
  python create_outdoor_segmentation.py \
    --image_path "wheat_inference_data/$IMAGE_NAME" \
    --yolo_weights weights/gwc_yolo_weights.pt \
    --sam_checkpoint weights/sam_vit_h_4b8939.pth \
    --output_dir outputs/colour_transfer_testing/segmentations
done