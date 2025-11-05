#!/bin/bash
# Quick baseline ERP QCM evaluation
# Evaluates the base model on ERP QCM dataset with all metrics

echo "=========================================="
echo "Baseline ERP QCM Evaluation"
echo "=========================================="

python3 evaluate_erp_qcm.py \
    --model-path "HuggingFaceTB/SmolVLM-500M-Instruct" \
    --dataset "dpo_image_dataset/qcm/qcm_dataset.json" \
    --image-dir "dpo_image_dataset" \
    --output-file "baseline_erp_qcm_evaluation.json"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: baseline_erp_qcm_evaluation.json"
echo "=========================================="
