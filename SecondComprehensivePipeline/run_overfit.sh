#!/bin/bash
#SBATCH --job-name=overfit_test
#SBATCH --output=logs_overfit/slurm_%j_%a.out
#SBATCH --error=logs_overfit/slurm_%j_%a.err
#SBATCH --time=71:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Overfit test job - runs a single config from overfitindividual folder
# Usage:
#   Single config:  sbatch run_overfit.sh overfit_erp_qcm_gemini
#   Array job:      sbatch --array=0-26 run_overfit.sh

# Load modules (adjust for your cluster)
# module load gcc cuda python

# Activate virtual environment (adjust path as needed)
# source /path/to/venv/bin/activate

cd /home/david-lacour/Documents/smolvlm/smolVLM/SecondComprehensivePipeline

# Create logs directory if it doesn't exist
mkdir -p logs_overfit

# List of all overfit configs
CONFIGS=(
    "overfit_erp_qcm_gemini"
    "overfit_erp_qcm_nova"
    "overfit_erp_qcm_procedure1"
    "overfit_erp_qcm_procedure2"
    "overfit_erp_dpo_gemini"
    "overfit_erp_dpo_nova"
    "overfit_erp_sft_chosen_rej_gemini"
    "overfit_erp_sft_chosen_rej_nova"
    "overfit_full_ft_qcm_gemini"
    "overfit_full_ft_qcm_nova"
    "overfit_full_ft_qcm_procedure1"
    "overfit_full_ft_qcm_procedure2"
    "overfit_full_ft_sft_chosen_rej_gemini"
    "overfit_full_ft_sft_chosen_rej_nova"
    "overfit_dpo_qcm_gemini"
    "overfit_dpo_qcm_nova"
    "overfit_dpo_qcm_procedure1"
    "overfit_dpo_qcm_procedure2"
    "overfit_sft_docvqa"
    "overfit_sft_ocrbench"
    "overfit_sft_chartqa"
    "overfit_full_ft_docvqa"
    "overfit_full_ft_ocrbench"
    "overfit_full_ft_chartqa"
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
    exit 1
fi

CONFIG_FILE="config/overfitindividual/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Running overfit test: $CONFIG_NAME"
echo "Config: $CONFIG_FILE"
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
