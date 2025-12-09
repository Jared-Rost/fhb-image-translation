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

# Run the SDXL inference script
python inference_test_sdxl.py \
  --num_images 20 \
  --output_dir outputs/v1_sdxl/test3 \
  --conditioning_image wheat_training_data/segmentation_masks_resized_1024/178149_25R61_1_3_10DAI_segmentation.png \
  --base_model_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_weights_path controlnet_training_output/v1_sdxl