# TextVQA Removal from Pipelines

## Date
2025-10-27

## Reason for Removal
TextVQA benchmark has been removed from all training pipelines because:
1. The TextVQA dataset (VQAv2) is 13.5GB and has extremely slow streaming (0.1 samples/s)
2. For practical training (>50 samples), it automatically falls back to using DocVQA dataset
3. This creates redundancy - training on "textvqa" would effectively train on DocVQA twice
4. Keeping it would waste computation time without adding value

## Files Modified

### 1. `run_systematic_benchmark_pipeline.py`
- **Line 52**: Changed `BENCHMARKS` from 4 to 3 benchmarks
- Removed "textvqa" from benchmark list: `["ocrbench", "docvqa", "chartqa"]`
- **Line 511**: Removed "textvqa" from `--train-benchmark` argument choices

### 2. `run_comprehensive_pipeline.py`
- **Lines 54-57**: Updated both `BENCHMARKS_TO_TRAIN` and `ALL_BENCHMARKS`
- Removed "textvqa" from both lists
- Added explanatory comment

### 3. `run_full_training_comparison.py`
- **Line 439-441**: Removed "textvqa" from `--benchmark-subset` choices
- Added explanatory note in help text

## Current Benchmark Configuration
All pipelines now use these **3 core benchmarks**:
1. **OCRBench** - OCR-focused tasks
2. **DocVQA** - Document question answering
3. **ChartQA** - Chart/graph understanding

## Implementation Details in finetune_on_benchmarks.py
The TextVQA training implementation remains in `finetune_on_benchmarks.py` but:
- For ≤50 samples: Uses VQAv2 streaming (acceptable wait time)
- For >50 samples: Automatically falls back to DocVQA with clear warning message
- This fallback behavior is why it was removed from pipelines

## Benefits
- ✅ Eliminates redundant training
- ✅ Saves computation time (~1 hour per pipeline run)
- ✅ Maintains dataset diversity (OCR, Documents, Charts)
- ✅ No loss of capability (DocVQA provides similar VQA training)
