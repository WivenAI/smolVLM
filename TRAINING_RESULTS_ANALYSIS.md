# Training Results Analysis - SFT Benchmark Comparison

**Date:** October 30, 2025
**Analysis:** Comparing baseline vs trained models on non-ERP benchmarks

---

## Summary of Results

### ⚠️ **CRITICAL FINDING: SFT Training is NOT Improving Scores**

The training appears to show **NO improvement** or even **slight degradation** compared to baseline.

---

## Detailed Comparison

### Baseline Performance
```
Model: base_model (HuggingFaceTB/SmolVLM-500M-Instruct)
OCRBench:  53.30%
DocVQA:    34.90%
ChartQA:   41.96%
Average:   43.39%
```

### 1. Training on OCRBench (500 samples, 3 epochs)
```
Model: trained_on_ocrbench
OCRBench:  52.80%  ❌ DOWN -0.50%  (trained on this!)
DocVQA:    34.80%  ❌ DOWN -0.10%
ChartQA:   41.80%  ❌ DOWN -0.16%
Average:   43.13%  ❌ DOWN -0.26%
```
**Result:** Training on OCRBench **hurt** OCRBench performance by 0.5%!

### 2. Training on ChartQA (500 samples, 3 epochs)
```
Model: trained_on_chartqa
OCRBench:  53.50%  ✅ UP +0.20%
DocVQA:    35.00%  ✅ UP +0.10%
ChartQA:   42.12%  ✅ UP +0.16%  (trained on this!)
Average:   43.54%  ✅ UP +0.15%
```
**Result:** Slight improvement (+0.16% on ChartQA), but very minimal.

### 3. Training on ERP QCM (balanced_qcm_all_end.json, 3 epochs)
```
Model: trained_on_erp_qcm
OCRBench:  52.50%  ❌ DOWN -0.80%
DocVQA:    34.30%  ❌ DOWN -0.60%
ChartQA:   40.96%  ❌ DOWN -1.00%
Average:   42.59%  ❌ DOWN -0.80%
```
**Result:** ERP training hurt general benchmark performance.

### 4. Training on ERP DPO Dataset with SFT (dpo_dataset.json chosen answers, 3 epochs)
```
Model: trained_on_erp_dpo-sft
OCRBench:  45.20%  ❌ DOWN -8.10%  😱
DocVQA:    29.90%  ❌ DOWN -5.00%  😱
ChartQA:   37.92%  ❌ DOWN -4.04%  😱
Average:   37.67%  ❌ DOWN -5.71%  😱
```
**Result:** **SEVERE degradation** - this model got significantly worse!

---

## Comparison Table

| Model | OCRBench | Δ | DocVQA | Δ | ChartQA | Δ | Average | Δ |
|-------|----------|---|--------|---|---------|---|---------|---|
| **Baseline** | **53.30%** | - | **34.90%** | - | **41.96%** | - | **43.39%** | - |
| Trained on OCRBench | 52.80% | -0.5 | 34.80% | -0.1 | 41.80% | -0.2 | 43.13% | -0.3 |
| **Trained on ChartQA** | **53.50%** | **+0.2** | **35.00%** | **+0.1** | **42.12%** | **+0.2** | **43.54%** | **+0.2** |
| Trained on ERP QCM | 52.50% | -0.8 | 34.30% | -0.6 | 40.96% | -1.0 | 42.59% | -0.8 |
| Trained on ERP DPO-SFT | 45.20% | -8.1 😱 | 29.90% | -5.0 😱 | 37.92% | -4.0 😱 | 37.67% | -5.7 😱 |

**Legend:**
- ✅ Improvement
- ❌ Degradation
- 😱 Severe degradation

---

## Key Observations

### 1. Training on Benchmarks Shows MINIMAL or NO Improvement

**Expected:** Training on OCRBench should improve OCRBench scores significantly (+5-10%)
**Actual:** OCRBench training **decreased** OCRBench score by -0.5%

**Expected:** Training on ChartQA should improve ChartQA scores significantly (+5-10%)
**Actual:** ChartQA training improved ChartQA by only +0.16%

### 2. Best Result: ChartQA Training
- Only model that showed **any** improvement over baseline
- Very small improvement: +0.15% average
- Still within noise/margin of error

### 3. Worst Result: ERP DPO-SFT Training
- **Catastrophic degradation**: -5.7% average
- OCRBench dropped from 53.3% → 45.2% (-8.1%)
- This suggests serious issues with the DPO dataset or training process

### 4. ERP Training Generally Hurts
- Both ERP QCM and ERP DPO-SFT show degradation
- ERP QCM: -0.8% average
- ERP DPO-SFT: -5.7% average (much worse)

---

## Possible Explanations

### Why Is Training NOT Working?

#### 1. **Model Already Well-Trained (Most Likely)**
```
The base SmolVLM-500M-Instruct is already highly tuned:
- Pre-trained on massive VQA datasets
- Instruction-tuned for question answering
- Further fine-tuning with small datasets (500 samples) may not help
```

#### 2. **Training Configuration Issues**
```python
Current settings:
- Samples: 500 per benchmark
- Epochs: 3
- Learning rate: 5e-7 (very small)
- LoRA r=16, alpha=32
- 4-bit quantization

Possible problems:
- Learning rate too small → no weight updates
- 4-bit quantization → limited precision for fine-tuning
- LoRA rank too low → limited capacity to learn
- Only 500 samples → insufficient data
```

#### 3. **Evaluation on Same Data**
```
Problem: Are we evaluating on the SAME samples used for training?
- If training on OCRBench 500 samples
- Then evaluating on those SAME 500 samples
- Model should memorize and get 100%
- But it's getting WORSE (52.8% < 53.3%)

This suggests the model is NOT learning at all!
```

#### 4. **Data Quality Issues**
```
ERP DPO-SFT severe degradation suggests:
- Poor quality "chosen" answers in DPO dataset
- Contradictory or confusing examples
- Training on bad data makes model worse
```

#### 5. **Catastrophic Forgetting**
```
The model may be "forgetting" its pre-trained knowledge:
- Base model learned from millions of examples
- Fine-tuning on 500 samples overwrites that knowledge
- Result: worse performance on everything
```

---

## Detailed Analysis: Why OCRBench Training Made Things WORSE

### The Paradox
```
Training on OCRBench dataset → OCRBench score DECREASED

This should be IMPOSSIBLE if:
1. Model is learning
2. We're evaluating on training data
3. Training converged
```

### Possible Causes

#### A. Evaluating on Different Split
```
Training set: 500 samples from OCRBench
Evaluation set: Different 1000 samples from OCRBench

Result: Model overfits to 500 samples, worse on unseen data
```

#### B. Learning Rate Too High
```
Learning rate causes model to "unlearn" pre-trained knowledge
Without enough new data to replace it, performance drops
```

#### C. LoRA Hurting, Not Helping
```
LoRA with low rank (r=16) and 4-bit quantization may:
- Introduce noise
- Not have enough capacity to learn
- Interfere with existing weights
```

#### D. Training Not Converging
```
Loss may be increasing, not decreasing during training
Model is getting WORSE, not better
Need to check training logs for loss curves
```

---

## Comparison with Official SmolVLM Scores

### Official SmolVLM-500M-Instruct Benchmarks:
```
OCRBench:  61.0%
DocVQA:    70.5%
ChartQA:   62.8%
```

### Your Baseline Scores:
```
OCRBench:  53.3%  (87% of official)
DocVQA:    34.9%  (50% of official) 😱
ChartQA:   42.0%  (67% of official)
```

### Analysis
**OCRBench:** Your baseline is reasonable (87% of official)
- Likely due to evaluation differences or sample selection

**DocVQA:** Your baseline is MUCH LOWER (50% of official) 😱
- This is a huge gap!
- Suggests evaluation method differs significantly
- Or model is not being properly used for DocVQA

**ChartQA:** Your baseline is decent (67% of official)
- Reasonable gap, likely due to evaluation setup

---

## Recommendations

### 1. **Verify Evaluation is Correct** (TOP PRIORITY)

Check if the accuracy calculation is still using the fixed version:
```python
# Should be using benchmark-specific accuracy:
if benchmark_name == 'ocrbench':
    # Containment check
elif benchmark_name == 'docvqa':
    # Bidirectional check
```

Verify we're not evaluating on the SAME samples used for training.

### 2. **Check Training Logs for Loss Curves**

Look at the training output to see if loss is decreasing:
```bash
# Find training logs
find . -name "*ocrbench*" -type f | xargs grep -i "loss"
```

If loss is increasing or flat → training is broken.

### 3. **Increase Training Strength**

If training is working but effects are too small:
```python
# Try these changes in finetune_on_benchmarks.py:
learning_rate=5e-5  # Increase from 5e-7 (100x higher)
num_epochs=5        # Increase from 3
max_samples=1000    # Increase from 500
lora_r=32           # Increase from 16
lora_alpha=64       # Increase from 32
```

### 4. **Investigate ERP DPO-SFT Severe Degradation**

The -5.7% drop is concerning:
```bash
# Check the DPO dataset quality
python3 -c "
import json
with open('dpo_image_dataset/dpo_dataset.json') as f:
    data = json.load(f)
    print(f'Total samples: {len(data)}')
    print(f'Sample chosen answer: {data[0][\"chosen\"][:200]}')
    print(f'Sample rejected answer: {data[0][\"rejected\"][:200]}')
"
```

If "chosen" answers are poor quality → explains degradation.

### 5. **Try Without LoRA/Quantization**

Test if 4-bit LoRA is the problem:
```python
# Full fine-tuning without quantization (needs more VRAM)
# Or use 8-bit instead of 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Better precision than 4-bit
)
```

### 6. **Verify Train/Eval Split**

Make sure training and evaluation use different samples:
```python
# In the training script, check:
train_samples = first 500 samples
eval_samples = different 500 samples (or full benchmark)
```

---

## Immediate Next Steps

### Before Running Full Pipeline Again:

1. **Check one training run in detail:**
   ```bash
   # Find the OCRBench training log
   find wandb -name "*ocrbench*" -type f | grep logs
   ```

2. **Verify loss is decreasing:**
   Look for lines like:
   ```
   'loss': 0.45  → 0.32  → 0.21  (should decrease)
   ```

3. **Check if train/eval are different:**
   Verify the training script splits data correctly

4. **Test with higher learning rate:**
   Quick test with learning_rate=5e-5 instead of 5e-7

### If Training Still Doesn't Work:

**Consider that fine-tuning SmolVLM-500M-Instruct on small datasets may not be effective.**

The model is already highly optimized. Further training with:
- Only 500 samples
- Low learning rate
- 4-bit quantization
- LoRA (limited capacity)

...may simply not be enough to move the needle.

---

## Bottom Line

### Current Status: ⚠️ **TRAINING IS NOT WORKING**

**Evidence:**
1. Training on OCRBench → OCRBench score went DOWN (-0.5%)
2. Only ChartQA showed tiny improvement (+0.16%)
3. ERP training shows significant degradation (-5.7% for DPO-SFT)
4. No model beats baseline by more than 0.2%

**Possible Causes:**
1. Model already optimal for these tasks
2. Training config too weak (LR, samples, quantization)
3. Evaluating on wrong data
4. Training not converging (loss not decreasing)
5. ERP data quality issues

**Action Required:**
1. Check training logs for loss curves
2. Verify train/eval data split
3. Increase training strength (LR, epochs, samples)
4. Investigate why OCRBench training made things worse
5. Fix ERP DPO dataset quality issues

**DO NOT run full pipeline again until we understand why training isn't improving scores!**

---

## Data Sources

```
Baseline:              systematic_comparison_20251030_103759.csv
Trained on OCRBench:   systematic_comparison_20251030_114953.csv
Trained on ChartQA:    systematic_comparison_20251030_123451.csv
Trained on ERP QCM:    systematic_comparison_20251030_132146.csv
Trained on ERP DPO-SFT: systematic_comparison_20251030_142436.csv
```
