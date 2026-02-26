#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=dpo_batch4
#SBATCH --output=izarlogs/dpo_%j_%a.out
#SBATCH --error=izarlogs/dpo_%j_%a.err
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# DPO Training Script - 25ComprehensivePipeline with batch_size=4
# ============================================================================
# All DPO configs with batch_size=4 for stability
#
# Usage:
#   Single config:  sbatch run_dpo_individual.sh erp_dpo_gemini
#   Array job:      sbatch --array=0-13 run_dpo_individual.sh
#   Limited:        sbatch --array=0-13%5 run_dpo_individual.sh  # max 5 concurrent
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

# List of all 14 DPO configs (batch_size=4, wandb_project=25ComprehensivePipeline)
CONFIGS=(
    # ERP DPO (QLoRA)
    "erp_dpo_gemini"
    "erp_dpo_nova"
    # DPO QCM (QLoRA)
    "dpo_qcm_gemini"
    "dpo_qcm_nova"
    "dpo_qcm_procedure1"
    "dpo_qcm_procedure2"
    # ERP QCM then DPO / DPO then QCM (QLoRA)
    "erp_qcm_dpo_gemini"
    "erp_dpo_qcm_gemini"
    "erp_dpo_qcm_nova"
    # Benchmark DPO (QLoRA)
    "dpo_chartqa"
    "dpo_docvqa"
    "dpo_ocrbench"
    # Full Fine-tune DPO
    "full_ft_dpo_gemini"
    "full_ft_dpo_nova"
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
    echo "  sbatch run_dpo_individual.sh <config_name>"
    echo "  sbatch --array=0-13 run_dpo_individual.sh"
    echo ""
    echo "Available configs (0-13):"
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
echo "DPO Training (batch_size=4)"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "File: $CONFIG_FILE"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "WandB Project: 25ComprehensivePipeline"
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
