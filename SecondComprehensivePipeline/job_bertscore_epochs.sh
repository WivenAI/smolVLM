#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=bs_epochs
#SBATCH --output=izarlogs/bs_epochs_%j.txt
#SBATCH --error=izarlogs/bs_epochs_err_%j.txt
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1

# BERTScore Evaluation with Epoch Tracking and WandB Logging
#
# Evaluates models at multiple epochs on DPO datasets
# Uses 200 samples per dataset
# Logs beautiful graphs to WandB showing progression
#
# Usage:
#   sbatch job_bertscore_epochs.sh
#   sbatch job_bertscore_epochs.sh --num-samples 300
#   sbatch job_bertscore_epochs.sh --model erp_qcm_gemini
#   sbatch job_bertscore_epochs.sh --epochs 0,1,5,10,20,50,100
#   sbatch job_bertscore_epochs.sh --include-full-ft

cd /scratch/izar/dlacour/wiven7/smolvlm/smolVLM/SecondComprehensivePipeline

export MKL_THREADING_LAYER=GNU
export HF_HOME=/scratch/izar/dlacour/hf_cache
export TRANSFORMERS_CACHE=/scratch/izar/dlacour/hf_cache
export HF_DATASETS_CACHE=/scratch/izar/dlacour/hf_cache/datasets

# WandB settings (online mode for real-time graphs)
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}

mkdir -p $HF_HOME
mkdir -p izarlogs

EXTRA_ARGS="${@:-}"

echo "========================================"
echo "BERTScore Epoch Evaluation"
echo "========================================"
echo "Started: $(date)"
echo "Extra args: $EXTRA_ARGS"
echo "========================================"

# Default: 200 samples, QLoRA models, epochs 0,1,5,10,20,50,100
python run_bertscore_eval.py \
    --num-samples 200 \
    --wandb-project "BERTScore-Epoch-Eval" \
    $EXTRA_ARGS

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
