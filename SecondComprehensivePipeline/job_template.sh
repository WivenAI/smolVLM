#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=wiven
#SBATCH --output=izarlogs/output_%x_%j.txt
#SBATCH --error=izarlogs/error_%x_%j.txt
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --time 71:00:00
#SBATCH --gres=gpu:1

# This script accepts a config file path as the first argument
# Usage: sbatch job_template.sh config/individual/conf_strategy_name.yaml

CONFIG_FILE=${1:-config/conf.yaml}

cd /scratch/izar/dlacour/wiven7/smolvlm/smolVLM/SecondComprehensivePipeline

export MKL_THREADING_LAYER=GNU

# Set HuggingFace cache to scratch (more space)
export HF_HOME=/scratch/izar/dlacour/hf_cache
export TRANSFORMERS_CACHE=/scratch/izar/dlacour/hf_cache
export HF_DATASETS_CACHE=/scratch/izar/dlacour/hf_cache/datasets

# Create the directory if it doesn't exist
mkdir -p $HF_HOME

export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512
export WANDB_MODE=offline

echo "Running pipeline with config: $CONFIG_FILE"
python pipeline.py --config "$CONFIG_FILE"
