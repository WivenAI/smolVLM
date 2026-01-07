#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=overfit_batch16
#SBATCH --output=izarlogs/overfit_%j_%a.out
#SBATCH --error=izarlogs/overfit_%j_%a.err
#SBATCH --time=71:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Overfit Test Script for 32GB GPU with batch_size=16
# ============================================================================
# All configs updated for 32GB GPU with batch_size=16 (conservative for all
# training types: QLoRA SFT/DPO, Full FT SFT/DPO)
#
# Usage:
#   Single config:  sbatch run_overfit.sh overfit_erp_qcm_gemini
#   Array job:      sbatch --array=0-26 run_overfit.sh
#   All jobs:       sbatch --array=0-26%5 run_overfit.sh  # max 5 concurrent
# ============================================================================

# Change to project directory (update for your cluster)
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
export WANDB_API_KEY=1d6641b737cd13fe32a9371dd3780308fee23512
export WANDB_MODE=offline

# List of all 27 overfit configs (batch_size=16 for 32GB GPU)
CONFIGS=(
    # ERP QCM SFT (QLoRA)
    "overfit_erp_qcm_gemini"
    "overfit_erp_qcm_nova"
    "overfit_erp_qcm_procedure1"
    "overfit_erp_qcm_procedure2"
    # ERP DPO (QLoRA)
    "overfit_erp_dpo_gemini"
    "overfit_erp_dpo_nova"
    # ERP SFT Chosen/Rejected (QLoRA)
    "overfit_erp_sft_chosen_rej_gemini"
    "overfit_erp_sft_chosen_rej_nova"
    # Full Fine-tune QCM
    "overfit_full_ft_qcm_gemini"
    "overfit_full_ft_qcm_nova"
    "overfit_full_ft_qcm_procedure1"
    "overfit_full_ft_qcm_procedure2"
    # Full Fine-tune SFT Chosen/Rejected
    "overfit_full_ft_sft_chosen_rej_gemini"
    "overfit_full_ft_sft_chosen_rej_nova"
    # DPO QCM (QLoRA)
    "overfit_dpo_qcm_gemini"
    "overfit_dpo_qcm_nova"
    "overfit_dpo_qcm_procedure1"
    "overfit_dpo_qcm_procedure2"
    # Benchmark SFT (QLoRA)
    "overfit_sft_docvqa"
    "overfit_sft_ocrbench"
    "overfit_sft_chartqa"
    # Benchmark Full Fine-tune
    "overfit_full_ft_docvqa"
    "overfit_full_ft_ocrbench"
    "overfit_full_ft_chartqa"
    # Benchmark DPO (QLoRA)
    "overfit_dpo_docvqa"
    "overfit_dpo_ocrbench"
    "overfit_dpo_chartqa"
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
    echo "  sbatch run_overfit.sh <config_name>"
    echo "  sbatch --array=0-26 run_overfit.sh"
    echo ""
    echo "Available configs (0-26):"
    for i in "${!CONFIGS[@]}"; do
        echo "  $i: ${CONFIGS[$i]}"
    done
    exit 1
fi

CONFIG_FILE="config/overfitindividual/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Overfit Test (32GB GPU, batch_size=16)"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "File: $CONFIG_FILE"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
