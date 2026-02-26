#!/bin/bash
#SBATCH --account=master
#SBATCH --job-name=bs_all
#SBATCH --output=izarlogs/bs_all_%A_%a.txt
#SBATCH --error=izarlogs/bs_all_err_%A_%a.txt
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 5
#SBATCH --time 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-41

# BERTScore Evaluation for ALL Models in modelweights/
#
# Runs as a SLURM array job: one GPU job per model (+ baseline).
# Each task evaluates one model on both DPO datasets (gemini + nova).
#
# Usage:
#   sbatch job_bertscore_all_models.sh                  # All models, 200 samples
#   sbatch job_bertscore_all_models.sh 300              # All models, 300 samples
#   sbatch job_bertscore_all_models.sh 200 "0,1,5,10"   # Custom epochs
#
# To run only a subset (e.g., models 0-9):
#   sbatch --array=0-9 job_bertscore_all_models.sh

cd /scratch/izar/dlacour/wiven7/smolvlm/smolVLM/SecondComprehensivePipeline

export MKL_THREADING_LAYER=GNU
export HF_HOME=/scratch/izar/dlacour/hf_cache
export TRANSFORMERS_CACHE=/scratch/izar/dlacour/hf_cache
export HF_DATASETS_CACHE=/scratch/izar/dlacour/hf_cache/datasets
export WANDB_ENTITY=david-lacour-epfl
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}

mkdir -p $HF_HOME
mkdir -p izarlogs

NUM_SAMPLES=${1:-200}
EPOCHS=${2:-"0,1,5,10,20,50,100"}

# List of ALL models (index 0 = baseline, 1-41 = modelweights)
MODELS=(
    "baseline"
    "dpo_chartqa"
    "dpo_docvqa"
    "dpo_ocrbench"
    "dpo_qcm_gemini"
    "dpo_qcm_nova"
    "dpo_qcm_procedure1"
    "dpo_qcm_procedure2"
    "erp_dpo_gemini"
    "erp_dpo_nova"
    "erp_dpo_qcm_gemini"
    "erp_dpo_qcm_nova"
    "erp_qcm_combined"
    "erp_qcm_dpo_gemini"
    "erp_qcm_gemini"
    "erp_qcm_nova"
    "erp_qcm_procedure1"
    "erp_qcm_procedure2"
    "erp_qcm_procedure_combined"
    "erp_qcm_sft_chosen_rej_combined"
    "erp_qcm_sft_chosen_rej_gemini"
    "erp_qcm_sft_chosen_rej_nova"
    "erp_sft_chosen_rej_gemini"
    "erp_sft_chosen_rej_nova"
    "erp_sft_chosen_rej_qcm_combined"
    "erp_sft_chosen_rej_qcm_gemini"
    "erp_sft_chosen_rej_qcm_nova"
    "full_ft_chartqa"
    "full_ft_docvqa"
    "full_ft_dpo_gemini"
    "full_ft_dpo_nova"
    "full_ft_ocrbench"
    "full_ft_qcm_combined"
    "full_ft_qcm_gemini"
    "full_ft_qcm_nova"
    "full_ft_qcm_procedure1"
    "full_ft_qcm_procedure2"
    "full_ft_sft_chosen_rej_gemini"
    "full_ft_sft_chosen_rej_nova"
    "sft_chartqa"
    "sft_docvqa"
    "sft_ocrbench"
)

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "BERTScore Evaluation - ALL MODELS"
echo "========================================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL_NAME"
echo "Samples: $NUM_SAMPLES"
echo "Epochs: $EPOCHS"
echo "Started: $(date)"
echo "========================================"

if [ "$MODEL_NAME" == "baseline" ]; then
    # Baseline: use pipeline.py with the baseline config
    echo "Evaluating BASELINE model (SmolVLM2-256M-Video-Instruct)"
    python pipeline.py --config config/individual/conf_bertscore_baseline.yaml
else
    # Fine-tuned model: use run_bertscore_eval.py
    echo "Evaluating fine-tuned model: $MODEL_NAME"
    python run_bertscore_eval.py \
        --model "$MODEL_NAME" \
        --num-samples "$NUM_SAMPLES" \
        --epochs "$EPOCHS" \
        --include-full-ft \
        --wandb-project "BERTScore-All-Models"
fi

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
