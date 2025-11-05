# ERP Evaluation Guide (QCM & DPO)

## Overview

This guide describes the comprehensive ERP evaluation system that measures model performance on ERP-specific tasks using **two datasets** with different metrics:

### ERP QCM Dataset (Multiple Choice Questions)
1. **Accuracy** - Does the model select the correct answer?
2. **BERTScore** - Semantic similarity between model response and correct answer
3. **Log Probability** - Confidence the model assigns to its answer

### ERP DPO Dataset (Direct Preference Optimization)
1. **Preference Accuracy** - Does the model prefer chosen over rejected response?
2. **Log Probability Margin** - Difference between chosen and rejected log probabilities
3. **BERTScore** - Semantic similarity between generated and chosen response

## What Changed

### New Files

1. **`evaluate_erp_qcm.py`** - Standalone evaluation script for ERP QCM dataset
   - Calculates accuracy, BERTScore, and log probabilities
   - Supports both nested and flat QCM dataset formats
   - Works with images from ERP interface screenshots

2. **`evaluate_erp_dpo.py`** - Standalone evaluation script for ERP DPO dataset
   - Calculates preference accuracy and log probability margins
   - Measures BERTScore between generated and chosen responses
   - Evaluates model's preference alignment

3. **`run_baseline_erp_eval.sh`** - Quick script to evaluate base model on QCM

### Modified Files

1. **`run_systematic_benchmark_pipeline.py`** - Integrated ERP QCM and DPO evaluation
   - New method: `evaluate_erp_qcm()` - Evaluates models on ERP QCM
   - New method: `evaluate_erp_dpo()` - Evaluates models on ERP DPO
   - Modified: `benchmark_model()` - Now runs both ERP QCM and DPO evaluations
   - Modified: `phase4_comparison()` - Includes ERP QCM and DPO metrics in comparison
   - Modified: `print_insights()` - Shows both ERP QCM and DPO performance insights
   - New argument: `--skip-erp-eval` - Skip all ERP evaluations (QCM and DPO)

## Usage

### Standalone ERP QCM Evaluation

Evaluate any model on the ERP QCM dataset:

```bash
python3 evaluate_erp_qcm.py \
    --model-path "HuggingFaceTB/SmolVLM-500M-Instruct" \
    --dataset "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
    --image-dir "dpo_image_dataset" \
    --output-file "results.json"
```

**Options:**
- `--model-path`: Path to model (base or fine-tuned)
- `--dataset`: Path to QCM dataset JSON
- `--image-dir`: Directory containing images
- `--output-file`: Where to save results
- `--max-samples`: Limit evaluation to N samples (for testing)
- `--no-bertscore`: Skip BERTScore calculation (faster)

### Standalone ERP DPO Evaluation

Evaluate any model on the ERP DPO dataset:

```bash
python3 evaluate_erp_dpo.py \
    --model-path "HuggingFaceTB/SmolVLM-500M-Instruct" \
    --dataset "dpo_image_dataset/dpo_dataset_cleaned.json" \
    --image-dir "dpo_image_dataset" \
    --output-file "results_dpo.json"
```

**Options:**
- `--model-path`: Path to model (base or fine-tuned)
- `--dataset`: Path to DPO dataset JSON
- `--image-dir`: Directory containing images
- `--output-file`: Where to save results
- `--max-samples`: Limit evaluation to N samples (for testing)
- `--no-bertscore`: Skip BERTScore calculation (faster)

### Baseline Evaluation

Quick baseline QCM evaluation:

```bash
./run_baseline_erp_eval.sh
```

This evaluates the base model on the full ERP QCM dataset.

For DPO evaluation, run:

```bash
python3 evaluate_erp_dpo.py --output-file baseline_erp_dpo_evaluation.json
```

### Full Pipeline with ERP Evaluation

The systematic pipeline now automatically evaluates on **both ERP QCM and DPO**:

```bash
# Baseline only (includes ERP QCM + DPO)
python3 run_systematic_benchmark_pipeline.py

# Train on ERP and evaluate (includes ERP QCM + DPO)
python3 run_systematic_benchmark_pipeline.py \
    --train-erp \
    --erp-strategy qcm \
    --qcm-dataset "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
    --dpo-dataset "dpo_image_dataset/dpo_dataset_cleaned.json" \
    --image-dir "dpo_image_dataset"

# Skip all ERP evaluations if needed
python3 run_systematic_benchmark_pipeline.py --skip-erp-eval
```

### Comprehensive Pipeline

The comprehensive pipeline also includes ERP QCM and DPO evaluation:

```bash
python3 run_comprehensive_pipeline.py \
    --qcm-datasets "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
    --dpo-datasets "dpo_image_dataset/dpo_dataset_cleaned.json" \
    --image-dir "dpo_image_dataset"
```

This will automatically evaluate all models (base + trained) on both ERP QCM and ERP DPO datasets.

## Output Format

### ERP QCM Evaluation Results

```json
{
  "model_path": "HuggingFaceTB/SmolVLM-500M-Instruct",
  "dataset_path": "dpo_image_dataset/qcm/qcm_dataset_gemini.json",
  "num_samples": 3680,
  "metrics": {
    "accuracy": 45.32,
    "avg_log_prob": -0.2156,
    "bertscore": {
      "precision": 0.7234,
      "recall": 0.6891,
      "f1": 0.7058
    }
  },
  "detailed_results": [
    {
      "id": 0,
      "question": "...",
      "correct_answer": "D",
      "predicted_answer": "D",
      "response": "D.",
      "is_correct": true,
      "avg_log_prob": -0.1523,
      "image_name": "image_001.png"
    }
  ]
}
```

### ERP DPO Evaluation Results

```json
{
  "model_path": "HuggingFaceTB/SmolVLM-500M-Instruct",
  "dataset_path": "dpo_image_dataset/dpo_dataset_cleaned.json",
  "num_samples": 1840,
  "metrics": {
    "avg_chosen_logprob": -5.234,
    "avg_rejected_logprob": -5.789,
    "avg_margin": 0.555,
    "preference_accuracy": 62.5,
    "bertscore": {
      "precision": 0.7123,
      "recall": 0.6645,
      "f1": 0.6875
    }
  },
  "detailed_results": [
    {
      "prompt": "Que montre cette image...",
      "chosen": "L'image présente une interface...",
      "rejected": "L'image montre un écran...",
      "generated": "Interface de configuration...",
      "chosen_avg_logprob": -5.123,
      "rejected_avg_logprob": -5.912,
      "margin": 0.789,
      "prefers_chosen": true,
      "image_name": "image_001.png",
      "type": "descriptive"
    }
  ]
}
```

### Pipeline Comparison Output

The systematic pipeline now includes ERP QCM and DPO columns in comparison tables:

```
model                average_accuracy  ocrbench_acc  docvqa_acc  chartqa_acc  erp_qcm_acc  erp_qcm_log_prob  erp_qcm_bertscore_f1  erp_dpo_pref_acc  erp_dpo_margin  erp_dpo_bertscore_f1
base_model           67.45             72.3          65.8        64.2         45.3         -0.2156           0.7058                62.5              0.555            0.6875
trained_on_erp_qcm   66.89             71.8          64.5        64.3         68.7         -0.1234           0.8234                75.3              1.234            0.7891
trained_on_erp_dpo   67.12             72.0          65.1        64.2         62.4         -0.1567           0.7891                82.1              1.567            0.8234
```

## Metrics Interpretation

### QCM Metrics

#### Accuracy
- **Range**: 0-100%
- **Higher is better**
- Measures if the model selects the correct multiple choice answer
- Random guessing = 25% (4 options)

#### Average Log Probability (QCM)
- **Range**: Typically -0.1 to -1.0 (closer to 0 is better)
- **Higher (less negative) is better**
- Measures model confidence in its answers
- More confident models have higher log probabilities

#### BERTScore F1 (QCM)
- **Range**: 0-1
- **Higher is better**
- Measures semantic similarity between response and correct answer
- Useful even when exact answer doesn't match
- F1 combines precision and recall

### DPO Metrics

#### Preference Accuracy
- **Range**: 0-100%
- **Higher is better**
- Measures how often model assigns higher probability to chosen over rejected
- Random = 50%
- Well-aligned model should be >70%

#### Log Probability Margin
- **Range**: Can be negative or positive
- **Higher (more positive) is better**
- Difference between chosen and rejected log probabilities
- Positive margin = model prefers chosen
- Negative margin = model prefers rejected (bad alignment)

#### BERTScore F1 (DPO)
- **Range**: 0-1
- **Higher is better**
- Measures semantic similarity between generated response and chosen response
- Shows how well model output matches preferred responses

## Expected Results

### Baseline Model (Untrained)

**QCM Performance:**
- Accuracy: ~25-35% (better than random, but not trained)
- Log Prob: ~-0.3 to -0.5
- BERTScore F1: ~0.6-0.7

**DPO Performance:**
- Preference Accuracy: ~50-60% (slightly better than random)
- Margin: ~0.0 to 0.5 (weak preference)
- BERTScore F1: ~0.6-0.7

### After ERP QCM Training (SFT)

**QCM Performance:**
- Accuracy: ~60-75% (significant improvement)
- Log Prob: ~-0.1 to -0.2 (more confident)
- BERTScore F1: ~0.75-0.85

**DPO Performance:**
- Preference Accuracy: ~65-75%
- Margin: ~0.8 to 1.2
- BERTScore F1: ~0.75-0.85

### After ERP DPO Training

**QCM Performance:**
- Accuracy: ~55-70%
- Log Prob: ~-0.15 to -0.25
- BERTScore F1: ~0.7-0.8

**DPO Performance:**
- Preference Accuracy: ~75-85% (strong alignment)
- Margin: ~1.2 to 2.0 (clear preference for chosen)
- BERTScore F1: ~0.8-0.9

## Key Insights Display

The pipeline now shows ERP-specific insights for both QCM and DPO:

```
🏢 ERP QCM Performance:
   Baseline:
      Accuracy:       45.32%
      Avg Log Prob:   -0.2156
      BERTScore F1:   0.7058

   Trained Models:

   📈 trained_on_erp_qcm:
      Accuracy:       68.74% (+23.42%)
      Avg Log Prob:   -0.1234 (+0.0922)
      BERTScore F1:   0.8234 (+0.1176)

   📈 trained_on_erp_dpo:
      Accuracy:       62.41% (+17.09%)
      Avg Log Prob:   -0.1567 (+0.0589)
      BERTScore F1:   0.7891 (+0.0833)

🎯 ERP DPO Performance:
   Baseline:
      Preference Accuracy: 62.50%
      Margin:              0.555
      BERTScore F1:        0.6875

   Trained Models:

   📈 trained_on_erp_qcm:
      Preference Accuracy: 75.34% (+12.84%)
      Margin:              1.234 (+0.679)
      BERTScore F1:        0.7891 (+0.1016)

   📈 trained_on_erp_dpo:
      Preference Accuracy: 82.12% (+19.62%)
      Margin:              1.567 (+1.012)
      BERTScore F1:        0.8234 (+0.1359)
```

## Testing

Quick test on 2 samples for QCM:

```bash
python3 evaluate_erp_qcm.py --max-samples 2 --output-file test_qcm_results.json
```

Quick test on 2 samples for DPO:

```bash
python3 evaluate_erp_dpo.py --max-samples 2 --output-file test_dpo_results.json
```

Each should complete in ~5-10 seconds and verify the setup is working.

## Dataset Support

The evaluation supports both QCM dataset formats:

1. **Nested format** (from `dpo_image_dataset/qcm/qcm_dataset_gemini.json`):
   ```json
   {
     "image_name": "image_001.png",
     "type": "qcm",
     "qcm": {
       "question": "...",
       "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
       "correct_answer": "D",
       "explanation": "..."
     }
   }
   ```

2. **Flat format** (from `balanced_qcm_all_end.json`):
   ```json
   {
     "id": 1,
     "question": "...",
     "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
     "correct_answer": "A"
   }
   ```

Both formats are automatically detected and handled.

## Requirements

The evaluation requires:
- `transformers`
- `torch`
- `PIL` (Pillow)
- `bert_score` (for BERTScore calculation)
- `tqdm` (for progress bars)

All dependencies are already included in the existing environment.

## Next Steps

To run a full comparison:

1. **Evaluate baseline on both QCM and DPO**:
   ```bash
   # QCM evaluation
   ./run_baseline_erp_eval.sh

   # DPO evaluation
   python3 evaluate_erp_dpo.py --output-file baseline_erp_dpo_evaluation.json
   ```

2. **Train on ERP QCM (SFT)**:
   ```bash
   python3 run_systematic_benchmark_pipeline.py \
       --train-erp \
       --erp-strategy qcm \
       --qcm-dataset "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
       --dpo-dataset "dpo_image_dataset/dpo_dataset_cleaned.json" \
       --image-dir "dpo_image_dataset"
   ```

3. **Train on ERP DPO**:
   ```bash
   python3 run_systematic_benchmark_pipeline.py \
       --train-erp \
       --erp-strategy dpo \
       --dpo-dataset "dpo_image_dataset/dpo_dataset_cleaned.json" \
       --qcm-dataset "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
       --image-dir "dpo_image_dataset"
   ```

4. **Compare results** - The pipeline automatically generates comparison tables with all metrics

5. **Run comprehensive pipeline** - For full systematic comparison of all strategies:
   ```bash
   python3 run_comprehensive_pipeline.py \
       --qcm-datasets "dpo_image_dataset/qcm/qcm_dataset_gemini.json" \
       --dpo-datasets "dpo_image_dataset/dpo_dataset_cleaned.json" \
       --image-dir "dpo_image_dataset"
   ```

This will:
- Evaluate base model on all benchmarks + ERP QCM + ERP DPO
- Train multiple variants (SFT on benchmarks, SFT on QCM, DPO, Combined)
- Generate comprehensive comparisons including all ERP metrics
- Show detailed insights on which training strategy works best
