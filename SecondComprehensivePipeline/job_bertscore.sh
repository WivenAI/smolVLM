#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=wiven_bertscore
#SBATCH --output=izarlogs/output_%x_%j.txt
#SBATCH --error=izarlogs/error_%x_%j.txt
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1

# BERTScore evaluation job
# Usage:
#   sbatch job_bertscore.sh <config_file>
#
# Available configs:
#   - conf_bertscore_baseline.yaml        : Baseline SmolVLM2-256M-Video-Instruct
#   - conf_bertscore_dpo_gemini.yaml      : erp_dpo_gemini/epoch_200_eval
#   - conf_bertscore_dpo_nova.yaml        : erp_dpo_nova/epoch_200_eval
#   - conf_bertscore_sft_chosen_gemini.yaml : erp_sft_chosen_rej_gemini/epoch_30_eval
#   - conf_bertscore_sft_chosen_nova.yaml   : erp_sft_chosen_rej_nova/epoch_30_eval
#
# Run all 5:
#   sbatch job_bertscore.sh config/individual/conf_bertscore_baseline.yaml
#   sbatch job_bertscore.sh config/individual/conf_bertscore_dpo_gemini.yaml
#   sbatch job_bertscore.sh config/individual/conf_bertscore_dpo_nova.yaml
#   sbatch job_bertscore.sh config/individual/conf_bertscore_sft_chosen_gemini.yaml
#   sbatch job_bertscore.sh config/individual/conf_bertscore_sft_chosen_nova.yaml

CONFIG_FILE=${1:-config/individual/conf_bertscore_dpo_gemini.yaml}

cd /scratch/izar/dlacour/wiven7/smolvlm/smolVLM/SecondComprehensivePipeline

export MKL_THREADING_LAYER=GNU

# Set HuggingFace cache to scratch (more space)
export HF_HOME=/scratch/izar/dlacour/hf_cache
export TRANSFORMERS_CACHE=/scratch/izar/dlacour/hf_cache
export HF_DATASETS_CACHE=/scratch/izar/dlacour/hf_cache/datasets

# Create the directory if it doesn't exist
mkdir -p $HF_HOME

export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export WANDB_MODE=offline

echo "Running BERTScore evaluation with config: $CONFIG_FILE"
python pipeline.py --config "$CONFIG_FILE"
