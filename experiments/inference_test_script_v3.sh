#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=01-00:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/env/wheat_infection_filter_env/bin/activate

# Run the updated inference script
python inference_test_v3.py \
  --num_images 20 \
  --output_dir outputs/v6 \
  --conditioning_image wheat_training_data/segmentation_masks_resized/8_Ava_1_2_14DAI_segmentation.png \
  --base_model_path sd-legacy/stable-diffusion-v1-5 \
  --controlnet_weights_path controlnet_training_output/v6