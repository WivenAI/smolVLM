# Pipeline Verification Checklist ✅

**Date:** 2025-10-30 10:38
**Status:** Pipeline restarted with FIXED code

---

## ✅ Pre-Flight Checks (COMPLETED)

### 1. Code Fixes Verified
- ✅ `run_systematic_benchmark_pipeline.py` updated with fixed `calculate_accuracy()`
- ✅ Function now handles `ground_truth` as lists properly
- ✅ Function accepts `benchmark_name` parameter
- ✅ Caller passes `benchmark_name=benchmark_name` on line 150
- ✅ All changes committed and pushed to git

### 2. Test Validation
- ✅ Ran `test_accuracy_fix.py` on old data
- ✅ Confirmed fix works:
  - OCRBench: 3.10% → 53.30% ✅
  - DocVQA: 7.20% → 34.90% ✅
  - ChartQA: 2.08% → 41.96% ✅

### 3. Pipeline Status
- ✅ Old pipeline stopped (PID 7927 from 09:26)
- ✅ New pipeline started (PID 13163 at 10:37)
- ✅ Using freshly committed code with fixes
- ✅ Log files created:
  - `comprehensive_results/pipeline_log_20251030_103758.txt`
  - `systematic_results/systematic_log_20251030_103759.txt`

---

## 🔍 What to Verify When Pipeline Completes

### Expected Output in Comparison Files

**In `systematic_results/systematic_comparison_*.json`:**

The "metrics" section should show:

```json
{
  "base_model": {
    "metrics": {
      "ocrbench": {
        "accuracy": 53.30,  // ← Should be 50-60%, NOT 3%
        "num_samples": 1000
      },
      "docvqa": {
        "accuracy": 34.90,  // ← Should be 30-40%, NOT 7%
        "num_samples": 1000
      },
      "chartqa": {
        "accuracy": 41.96,  // ← Should be 40-50%, NOT 2%
        "num_samples": 2500
      }
    }
  }
}
```

**In `systematic_results/systematic_comparison_*.csv`:**

```csv
model,ocrbench_acc,docvqa_acc,chartqa_acc,average_accuracy
base_model,53.30,34.90,41.96,43.39
```

### ❌ If You See This (BROKEN - OLD CODE):

```csv
model,ocrbench_acc,docvqa_acc,chartqa_acc,average_accuracy
base_model,3.10,7.20,2.08,4.13
```

Then something went wrong and it's using cached old code.

---

## 🎯 Success Criteria

### ✅ PASS if:
1. **OCRBench accuracy:** 50-60% (not 3%)
2. **DocVQA accuracy:** 30-40% (not 7%)
3. **ChartQA accuracy:** 40-55% (not 2%)
4. **Average accuracy:** 40-50% (not 4%)

### ❌ FAIL if:
1. Any benchmark shows < 10% accuracy
2. Average accuracy is < 10%
3. Results match the old broken pattern (3%, 7%, 2%)

---

## 📊 What the Results Will Tell Us

Once the pipeline completes, you'll be able to see:

### 1. **Is the base model performing correctly?**
Expected: OCRBench ~53%, DocVQA ~35%, ChartQA ~42%

### 2. **Did training on OCRBench improve OCRBench?**
Compare:
- Base model OCRBench: ~53%
- Trained on OCRBench: ???%

**If trained > 58%:** ✅ Training works!
**If trained ≈ 53%:** ⚠️ Training not helping
**If trained < 50%:** ❌ Training hurts performance

### 3. **Which training strategy is best?**
The comparison will show:
- SFT on each benchmark
- ERP QCM training
- ERP DPO training
- Combined strategies

---

## 🔧 What the Pipeline Will Do

### Phase 1: Baseline (Currently Running)
- Evaluate base model on OCRBench, DocVQA, ChartQA
- Save to `systematic_results/base_model_*.json`
- ⏱️ **ETA:** ~30-45 minutes for 4500 samples

### Phase 2: Benchmark Training
- Train on DocVQA (500 samples, 3 epochs) → ~30 min
- Train on OCRBench (500 samples, 3 epochs) → ~30 min
- Train on ChartQA (500 samples, 3 epochs) → ~30 min
- Evaluate each on all 3 benchmarks → ~30 min each

### Phase 3-6: ERP Training
- ERP QCM (SFT) → ~45 min
- ERP DPO dataset with SFT → ~45 min
- ERP DPO method → ~60 min (or may fail again)
- ERP Combined (QCM+DPO) → ~90 min

### Phase 7: Mega Comparison
- Aggregate all results
- Generate comparison tables
- Show insights

**Total estimated time:** 6-8 hours

---

## 🚨 What to Monitor

### During Baseline (First 30-45 min)
Check: `tail -f systematic_results/systematic_log_20251030_103759.txt`

Look for:
```
COMPREHENSIVE COMPARISON RESULTS
================================================================================

              model  average_accuracy  ocrbench_acc  docvqa_acc  chartqa_acc
base_model              43.39            53.30         34.90        41.96
```

If you see **43.39%** average: ✅ **FIX IS WORKING!**
If you see **4.13%** average: ❌ **STILL BROKEN** - stop and investigate

### During Training
Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

Check training logs for loss decreasing:
```bash
tail -f systematic_results/trained_on_*/checkpoint-*/trainer_state.json
```

### When Complete
Check final comparison:
```bash
cat systematic_results/systematic_comparison_*.csv | tail -20
```

---

## 📝 Files to Check After Completion

1. **Comparison files:**
   - `systematic_results/systematic_comparison_*.csv`
   - `systematic_results/systematic_comparison_*.json`
   - `comprehensive_results/mega_comparison_*.csv`
   - `comprehensive_results/mega_comparison_*.json`

2. **Individual model results:**
   - `systematic_results/base_model_*.json`
   - `systematic_results/trained_on_ocrbench_*.json`
   - `systematic_results/trained_on_docvqa_*.json`
   - `systematic_results/trained_on_chartqa_*.json`

3. **Training checkpoints:**
   - `systematic_results/trained_on_*/checkpoint-*/trainer_state.json`

---

## ✅ Final Verification Steps

When pipeline completes:

```bash
# 1. Check comparison results
cat systematic_results/systematic_comparison_*.csv | tail -1

# 2. If average_accuracy > 40%, run:
python3 -c "
import json
from pathlib import Path

# Get latest comparison
files = sorted(Path('systematic_results').glob('systematic_comparison_*.json'))
if files:
    with open(files[-1]) as f:
        data = json.load(f)

    print('\\n✅ VERIFICATION RESULTS:\\n')
    for model, info in data.items():
        metrics = info.get('metrics', {})
        print(f'{model}:')
        for bench, vals in metrics.items():
            acc = vals.get('accuracy', 0)
            status = '✅' if acc > 30 else '❌'
            print(f'  {status} {bench}: {acc:.2f}%')
        print()
"

# 3. Compare base vs trained on OCRBench
python3 analyze_all_results.py
```

---

## 🎯 Success Confirmation

You'll know everything worked if:

1. ✅ Base model shows 50%+ on OCRBench
2. ✅ Base model shows 30%+ on DocVQA
3. ✅ Base model shows 40%+ on ChartQA
4. ✅ Comparison files show realistic accuracy values
5. ✅ Can see if training actually improved performance

---

## 🔄 Current Status

- **Pipeline Status:** ✅ Running with fixed code
- **Started:** 10:37:59 (Oct 30, 2025)
- **Current Phase:** Phase 1 - Baseline evaluation
- **Expected Completion:** ~16:00-18:00 (6-8 hours)

**Check progress:**
```bash
tail -f systematic_results/systematic_log_20251030_103759.txt
```

**Monitor GPU:**
```bash
watch -n 1 nvidia-smi
```

---

## 👍 Bottom Line

**Everything is set up correctly!**

The pipeline is now running with the fixed accuracy calculation code. When it completes, you'll finally be able to see:

1. ✅ **Correct baseline performance** (50-60% on OCRBench, not 3%)
2. ✅ **Whether training actually helps** (compare trained vs base)
3. ✅ **Which training strategy works best** (SFT vs DPO vs Combined)

Just let it run and check back in ~6 hours! 🚀
