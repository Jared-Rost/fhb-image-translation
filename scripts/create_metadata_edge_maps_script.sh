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

# Run the metadata creation script
python create_metadata_edge_maps.py \
  --excel_file ./wheat_training_data/Final_Single_Head_Data.xlsx \
  --images_dir ./wheat_training_data/database_single_head_images_resized_1024 \
  --conditioning_dir ./wheat_training_data/edge_maps_resized_1024 \
  --output_file ./wheat_training_data/metadata.jsonl