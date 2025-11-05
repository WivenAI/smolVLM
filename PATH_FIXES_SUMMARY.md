# Dataset Path Fixes Summary

## Issue
Many files were referencing `dpo_image_dataset/dpo_dataset.json` which doesn't exist at the root level.

## Correct Dataset Paths

### DPO Datasets:
- ✅ **`dpo_image_dataset/dpo_dataset_cleaned.json`** (1.9M) - **USE THIS** (most current)
- ✅ `dpo_image_dataset/dpo_dataset_nova_pro.json` (1.2M) - Nova Pro version
- ⚠️ `dpo_image_dataset/old/dpo_dataset.json` (16M) - Old archived version
- ⚠️ `dpo_image_dataset/qcm/dpo_dataset.json` (1.9M) - Copy in qcm subdirectory

### QCM Datasets:
- ✅ **`balanced_qcm_all_end.json`** (1099 samples) - Balanced QCM (text-only)
- ✅ `dpo_image_dataset/qcm/qcm_dataset_gemini.json` (3.0M) - QCM with images
- ✅ `dpo_image_dataset/qcm/qcm_dataset_nova_pro.json` (2.0M) - Nova Pro version

## Files Fixed (14 files)

### Training Scripts:
1. ✅ `finetune_smolvlm_lora.py`
2. ✅ `finetune_smolvlm_sft.py`

### Evaluation Scripts:
3. ✅ `evaluate_bertscore_dpo.py`
4. ✅ `evaluate_dpo_logprobs.py`

### Pipeline Scripts:
5. ✅ `run_systematic_benchmark_pipeline.py`
6. ✅ `benchmark_pipeline.py`
7. ✅ `run_full_training_comparison.py`

### Test Scripts:
8. ✅ `test_dpo_tokenization.py`
9. ✅ `test_dpo_sample_sizes.py`
10. ✅ `test_dpo_training_quick.py`
11. ✅ `test_bertscore.py`
12. ✅ `test_logprob.py`
13. ✅ `check_missing_images.py`
14. ✅ `inspect_sample_139.py`

## Changes Made

All occurrences of:
```python
"dpo_image_dataset/dpo_dataset.json"
```

Were changed to:
```python
"dpo_image_dataset/dpo_dataset_cleaned.json"
```

## Verification

Confirmed no remaining incorrect references:
```bash
grep -r "dpo_dataset\.json\"" --include="*.py" . | grep -v "dpo_dataset_cleaned" | grep -v "old/"
# Returns: (empty - all fixed!)
```

## Your Pipeline Should Now Work

The systematic pipeline will now correctly find the DPO dataset:

```bash
python3 run_systematic_benchmark_pipeline.py \
    --train-erp \
    --erp-strategy qcm \
    --qcm-dataset "balanced_qcm_all_end.json" \
    --dpo-dataset "dpo_image_dataset/dpo_dataset_cleaned.json" \
    --image-dir "dpo_image_dataset"
```

Or just use defaults (now correct):

```bash
python3 run_systematic_benchmark_pipeline.py --train-erp --erp-strategy qcm
```

## Status: ✅ READY TO RUN

All dataset paths have been corrected and verified!
