# DPO Evaluation Guide: BERTScore & LogProb

## Overview

For the ERP DPO column in your results table, you need **DPO LogProb** and **BERTScore** metrics instead of accuracy.

## Evaluation Scripts

### 1. **DPO Log Probability Evaluation**

**Script**: `evaluate_dpo_logprobs.py`

**Metrics computed**:
- `chosen_logprob_avg`: Average log probability for chosen responses
- `rejected_logprob_avg`: Average log probability for rejected responses
- `margin_avg`: Average margin (chosen - rejected logprobs)
- `preference_accuracy`: % of times chosen > rejected
- `chosen_perplexity_avg`: Average perplexity for chosen responses
- `rejected_perplexity_avg`: Average perplexity for rejected responses

**Usage**:
```bash
python3 evaluate_dpo_logprobs.py \
    --model-path <model_directory> \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output dpo_logprob_results_<model_name>.json
```

**Output**: JSON file with per-example and overall metrics

**Dataset size**: ALL samples (1833 samples)

---

### 2. **BERTScore Evaluation**

**Script**: `evaluate_bertscore_dpo.py`

**Metrics computed**:
- `precision_avg`: Average BERTScore precision
- `recall_avg`: Average BERTScore recall
- `f1_avg`: Average BERTScore F1

**Usage**:
```bash
python3 evaluate_bertscore_dpo.py \
    --model-path <model_directory> \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output bertscore_results_<model_name>.json
```

**Output**: JSON file with per-example and overall metrics

**Dataset size**: ALL samples (1833 samples)

---

## Models to Evaluate

Based on your table, you need to run these evaluations on:

1. **Base Model**: `HuggingFaceTB/SmolVLM-500M-Instruct`
2. **Trained on ERP QCM**: `./results/trained_on_erp_qcm/`
3. **Trained on DocVQA**: `./results/trained_on_docvqa/`
4. **Trained on ERP DPO-SFT**: `./results/trained_on_erp_dpo_sft/`
5. **Trained on ERP DPO**: `./results/trained_on_erp_dpo/`
6. **Trained on OCRBench**: `./results/trained_on_ocrbench/`
7. **Trained on ChartQA**: `./results/trained_on_chartqa/`
8. **Trained on ERP QCM+DPO**: `./results/trained_on_erp_qcm_dpo/`
9. **Trained on ERP QCM+DPO-SFT**: `./results/trained_on_erp_qcm_dpo_sft/` (NEW!)

---

## Example Workflow

```bash
# For each model, run both evaluations:

# 1. Base Model
python3 evaluate_dpo_logprobs.py \
    --model-path HuggingFaceTB/SmolVLM-500M-Instruct \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output results/dpo_logprob_base.json

python3 evaluate_bertscore_dpo.py \
    --model-path HuggingFaceTB/SmolVLM-500M-Instruct \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output results/bertscore_base.json

# 2. Trained on ERP QCM
python3 evaluate_dpo_logprobs.py \
    --model-path ./results/trained_on_erp_qcm/ \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output results/dpo_logprob_erp_qcm.json

python3 evaluate_bertscore_dpo.py \
    --model-path ./results/trained_on_erp_qcm/ \
    --dataset dpo_image_dataset/dpo_dataset_gemini.json \
    --image-dir dpo_image_dataset \
    --output results/bertscore_erp_qcm.json

# ... repeat for all models
```

---

## Updated Table Format

Your table should include these metrics for the ERP DPO column:

```
Model Configuration | ERP DPO LogProb | ERP DPO BERTScore F1
                   | (margin)        | (%)
-------------------|-----------------|---------------------
Base Model         | X.XX            | XX.X
Trained on ERP QCM | X.XX            | XX.X
...
```

**Key Metrics to Report**:
- **DPO LogProb**: Use `margin_avg` (higher is better)
- **BERTScore**: Use `f1_avg` (higher is better)

---

## Notes

- Both evaluations run on the **FULL dataset** (all 1833 samples)
- Results are saved as JSON files with detailed per-example metrics
- Each evaluation takes ~10-30 minutes depending on model and hardware
- Make sure models are in the correct paths before running evaluations
