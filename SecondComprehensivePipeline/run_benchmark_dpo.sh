#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=bench_dpo
#SBATCH --output=izarlogs/bench_dpo_%j_%a.out
#SBATCH --error=izarlogs/bench_dpo_%j_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Benchmark DPO Training Script
# ============================================================================
# Runs DPO training on benchmark datasets (ChartQA, DocVQA, OCRBench)
# with fixed learning rate (1e-6)
#
# Usage:
#   Single config:  sbatch run_benchmark_dpo.sh chartqa
#   Array job:      sbatch --array=0-5 run_benchmark_dpo.sh
#   Limited:        sbatch --array=0-5%3 run_benchmark_dpo.sh  # max 3 concurrent
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

# List of benchmark DPO configs (LoRA + Full Fine-tune)
CONFIGS=(
    # LoRA/QLoRA DPO on benchmarks
    "dpo_chartqa"
    "dpo_docvqa"
    "dpo_ocrbench"
    # Full Fine-tune DPO on benchmarks
    "full_ft_dpo_chartqa"
    "full_ft_dpo_docvqa"
    "full_ft_dpo_ocrbench"
)

# Determine which config to run
if [ -n "$1" ]; then
    # Config name passed as argument (allow short names)
    case "$1" in
        chartqa|dpo_chartqa)
            CONFIG_NAME="dpo_chartqa"
            ;;
        docvqa|dpo_docvqa)
            CONFIG_NAME="dpo_docvqa"
            ;;
        ocrbench|dpo_ocrbench)
            CONFIG_NAME="dpo_ocrbench"
            ;;
        *)
            CONFIG_NAME="$1"
            ;;
    esac
elif [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array job - use task ID to select config
    CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
else
    echo "Error: No config specified. Use:"
    echo "  sbatch run_benchmark_dpo.sh <config_name>"
    echo "  sbatch --array=0-5 run_benchmark_dpo.sh"
    echo ""
    echo "Available configs:"
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
echo "Benchmark DPO Training"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "File: $CONFIG_FILE"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "WandB Project: 25ComprehensivePipeline"
echo "Learning Rate: 1e-6 (fixed)"
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
