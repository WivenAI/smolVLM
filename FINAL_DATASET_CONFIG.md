# Final Dataset Configuration

## ✅ All Datasets Configured

### DPO Datasets (Both Will Be Used):
1. **`dpo_image_dataset/dpo_dataset_gemini.json`** (1.9M) - Gemini-generated DPO pairs
2. **`dpo_image_dataset/dpo_dataset_nova_pro.json`** (1.2M) - Nova Pro-generated DPO pairs

### QCM Datasets (All Available):
1. **`balanced_qcm_all_end.json`** (1099 samples) - Balanced QCM without images
2. **`dpo_image_dataset/qcm/qcm_dataset.json`** (3.0M) - Gemini QCM with images
3. **`dpo_image_dataset/qcm/qcm_dataset_nova_pro.json`** (2.0M) - Nova Pro QCM with images

## Default Configuration

### Systematic Pipeline (Single Run):
- Default QCM: `dpo_image_dataset/qcm/qcm_dataset.json` (Gemini with images)
- Default DPO: `dpo_image_dataset/dpo_dataset_gemini.json`
- Note: `balanced_qcm_all_end.json` is text-only, used for evaluation and end-of-study training

### Comprehensive Pipeline (Full Comparison):
- QCM Datasets: All 3 (Gemini, Nova Pro, Balanced)
- DPO Datasets: Both (Gemini and Nova Pro)

## How to Run

### Quick Single Evaluation (Gemini with images):
```bash
python3 run_systematic_benchmark_pipeline.py --train-erp --erp-strategy qcm
```

Uses defaults:
- QCM: `dpo_image_dataset/qcm/qcm_dataset.json` (Gemini image-based)
- DPO: `dpo_image_dataset/dpo_dataset_gemini.json`

### Full Comparison (Both Gemini and Nova):
```bash
python3 run_comprehensive_pipeline.py
```

Automatically uses:
- **QCM**: 3 datasets (Gemini, Nova Pro, Balanced)
- **DPO**: 2 datasets (Gemini, Nova Pro)

This will train and evaluate on ALL combinations!

### Custom Dataset Selection:
```bash
# Use specific datasets
python3 run_comprehensive_pipeline.py \
    --qcm-datasets "dpo_image_dataset/qcm/qcm_dataset.json" \
    --dpo-datasets "dpo_image_dataset/dpo_dataset_gemini.json" "dpo_image_dataset/dpo_dataset_nova_pro.json"
```

### Using Balanced QCM (Text-Only) for Evaluation or Final Training:
```bash
# For evaluation only (on already-trained models)
python3 evaluate_erp_qcm.py \
    --model-path ./smolvlm-qcm-finetuned \
    --dataset balanced_qcm_all_end.json

# For final comprehensive training + evaluation (text-only)
python3 run_systematic_benchmark_pipeline.py \
    --train-erp \
    --erp-strategy qcm \
    --qcm-dataset balanced_qcm_all_end.json
```

## Files Updated (All 14 Files)

All Python scripts now use `dpo_dataset_gemini.json` as the default:

### Training:
- ✅ `finetune_smolvlm_lora.py`
- ✅ `finetune_smolvlm_sft.py`

### Evaluation:
- ✅ `evaluate_erp_dpo.py`
- ✅ `evaluate_bertscore_dpo.py`
- ✅ `evaluate_dpo_logprobs.py`

### Pipelines:
- ✅ `run_systematic_benchmark_pipeline.py`
- ✅ `run_comprehensive_pipeline.py`
- ✅ `benchmark_pipeline.py`
- ✅ `run_full_training_comparison.py`

### Tests:
- ✅ `test_dpo_tokenization.py`
- ✅ `test_dpo_sample_sizes.py`
- ✅ `test_dpo_training_quick.py`
- ✅ `test_bertscore.py`
- ✅ `test_logprob.py`

## Your Study is Now Ready!

Run the comprehensive pipeline to train and evaluate on **BOTH Gemini and Nova Pro datasets**:

```bash
python3 run_comprehensive_pipeline.py --epochs 3
```

This will:
1. Evaluate base model on all benchmarks + **both Gemini and Nova DPO** + **all QCM datasets**
2. Train on:
   - Each benchmark (DocVQA, OCRBench, ChartQA)
   - **Gemini QCM** (SFT)
   - **Nova QCM** (SFT)
   - **Gemini DPO** (DPO method)
   - **Nova DPO** (DPO method)
   - Combined strategies
3. Generate comprehensive comparison tables showing which dataset works best!

## Expected Results Tables

You'll get comparison tables like:

```
model                    ocr  doc  chart  gemini_qcm  nova_qcm  gemini_dpo  nova_dpo
base_model               72%  66%  64%    45%         43%       63%         61%
trained_on_gemini_qcm    71%  65%  64%    69%         ??        ??          ??
trained_on_nova_qcm      71%  65%  64%    ??          71%       ??          ??
trained_on_gemini_dpo    72%  65%  64%    ??          ??        82%         ??
trained_on_nova_dpo      72%  65%  64%    ??          ??        ??          85%
```

This will answer: **Which dataset (Gemini vs Nova) produces better results?** 🎯
