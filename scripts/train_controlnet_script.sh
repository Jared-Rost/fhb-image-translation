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

source ~/env/wheat_infection_filter_env/bin/activate
export TORCH_HOME=~/projects/def-cjhuofw-ab/rostj/torch-cache

# Run the training script
python train_controlnet_modified.py \
  --pretrained_model_name_or_path="sd-legacy/stable-diffusion-v1-5" \
  --controlnet_model_name_or_path="lllyasviel/sd-controlnet-seg" \
  --train_data_dir="./wheat_training_data" \
  --image_column="file_name" \
  --conditioning_image_column="conditioning_image" \
  --caption_column="text" \
  --output_dir="./controlnet_training_output/v6" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=2 \
  --num_train_epochs=9 \
  --checkpointing_steps=300 \
  --resume_from_checkpoint="latest" \
  --mixed_precision="fp16" \
  --dataloader_num_workers=0 \
  --seed=42