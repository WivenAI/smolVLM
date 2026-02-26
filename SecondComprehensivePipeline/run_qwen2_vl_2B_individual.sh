#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=qwen2_vl_2B
#SBATCH --output=izarlogs/qwen2_vl_2B_%j_%a.out
#SBATCH --error=izarlogs/qwen2_vl_2B_%j_%a.err
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Qwen2-VL-2B Training Script - 23ComprehensivePipeline
# ============================================================================
# Model: Qwen/Qwen2-VL-2B-Instruct
#
# Usage:
#   Single config:  sbatch run_qwen2_vl_2B_individual.sh qwen2_vl_2B_full_sft_ocrbench
#   Array job:      sbatch --array=0-11 run_qwen2_vl_2B_individual.sh
#   Limited:        sbatch --array=0-11%5 run_qwen2_vl_2B_individual.sh  # max 5 concurrent
# ============================================================================

# Change to project directory
cd /scratch/izar/dlacour/wiven7/smolvlm/smolVLM/SecondComprehensivePipeline

# Create logs directory
mkdir -p izarlogs

# Environment setup
export MKL_THREADING_LAYER=GNU
export HF_HOME=/scratch/izar/dlacour/hf_cache
export TRANSFORMERS_CACHE=/scratch/izar/dlacour/hf_cache
export HF_DATASETS_CACHE=/scratch/izar/dlacour/hf_cache/datasets
mkdir -p $HF_HOME

# WandB setup (offline mode for cluster)
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export WANDB_MODE=offline

# List of all 12 Qwen2-VL-2B configs
CONFIGS=(
    # Full FT Benchmark (3)
    "qwen2_vl_2B_full_sft_ocrbench"
    "qwen2_vl_2B_full_sft_docvqa"
    "qwen2_vl_2B_full_sft_chartqa"
    # Full FT ERP (6)
    "qwen2_vl_2B_full_sft_chosen_nova"
    "qwen2_vl_2B_full_sft_chosen_gemini"
    "qwen2_vl_2B_full_sft_qcm_nova"
    "qwen2_vl_2B_full_sft_qcm_gemini"
    "qwen2_vl_2B_full_sft_qcm_procedure1"
    "qwen2_vl_2B_full_sft_qcm_procedure2"
    # QLoRA Benchmark (3)
    "qwen2_vl_2B_qlora_sft_ocrbench"
    "qwen2_vl_2B_qlora_sft_docvqa"
    "qwen2_vl_2B_qlora_sft_chartqa"
)

# Determine which config to run
if [ -n "$1" ]; then
    # Config name passed as argument
    CONFIG_NAME="$1"
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array job - use task ID to select config
    CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
else
    echo "Error: No config specified. Use:"
    echo "  sbatch run_qwen2_vl_2B_individual.sh <config_name>"
    echo "  sbatch --array=0-11 run_qwen2_vl_2B_individual.sh"
    echo ""
    echo "Available configs (0-11):"
    for i in "${!CONFIGS[@]}"; do
        echo "  $i: ${CONFIGS[$i]}"
    done
    exit 1
fi

CONFIG_FILE="config/individual/conf_${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Qwen2-VL-2B Training"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "File: $CONFIG_FILE"
echo "Model: Qwen/Qwen2-VL-2B-Instruct"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "WandB Project: 23ComprehensivePipeline"
echo "Started: $(date)"
echo "=========================================="

# Run the pipeline
python pipeline.py --config "$CONFIG_FILE"

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
