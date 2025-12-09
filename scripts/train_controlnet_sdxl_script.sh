#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --time=03-00:00:00
#SBATCH --partition=livi

export HF_HOME=~/projects/def-cjhuofw-ab/rostj/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

source $HOME/env/wheat_infection_filter_env/bin/activate

# Run SDXL ControlNet training with edge maps
python train_controlnet_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="./wheat_training_data" \
  --image_column="image" \
  --conditioning_image_column="conditioning_image" \
  --caption_column="text" \
  --output_dir="./controlnet_training_output/v1_sdxl" \
  --resolution=1024 \
  --learning_rate=1e-5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --num_train_epochs=15 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --resume_from_checkpoint="latest" \
  --validation_prompt="AINBN3tpDb, moderate Blight, 45.0%" \
  --validation_image="./wheat_training_data/segmentation_masks_resized_1024/177406_25R61_1_2_14DAI_segmentation.png" \
  --validation_steps=500 \
  --num_validation_images=0 \
  --report_to="tensorboard" \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --dataloader_num_workers=4 \
  --seed=42