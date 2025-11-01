# Critical Training Format Fix - Oct 31, 2025

## Problem Discovered

**Sanity Check Results:**
- Base model accuracy: **53.30%**
- After 1 epoch training: **16.40%** (❌ **-36.90%** degradation!)
- Training loss: 16.16 → 0.32 (✅ **-98%** improvement)

**Paradox:** Loss improved dramatically, but accuracy got MUCH worse!

## Root Cause

**Format Mismatch Between Training and Evaluation**

### OLD Training Format (BROKEN)
```python
# finetune_on_benchmarks.py (lines 196-197)
prompt = f"<image>{question}"
full_text = f"{prompt}\n{answer}"
```

### Evaluation Format
```python
# evaluate_ocrbench.py (lines 84-94)
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}]
    }
]
prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
```

### Impact
- Training: Model learned `<image>question\nanswer` format perfectly (loss ↓98%)
- Evaluation: Used chat template format the model NEVER saw
- Result: Model hallucinated wrong text from images:
  - Read "CHRIS" instead of "CHAIN"
  - Read "C" instead of "CLOSE"

## Fix Applied

### NEW Training Format (FIXED)
```python
# finetune_on_benchmarks.py (lines 195-207)
# Format using chat template (MUST match evaluation format!)
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}]
    },
    {
        "role": "assistant",
        "content": answer
    }
]
full_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
```

## Verification

✅ **Test passed:** Training with 10 samples completed successfully
- No errors
- Format matches evaluation
- Ready for full-scale training

## Next Steps

1. **Re-train on 1000 OCRBench samples** with corrected format
2. **Re-run sanity check** to verify accuracy improves
3. **Expected result:**
   - Loss: ↓ (should still decrease)
   - Accuracy: ↑ (should INCREASE this time!)

## Key Lesson

**Always ensure training and evaluation use IDENTICAL formatting!**
- Use `processor.apply_chat_template()` for both
- Never mix custom formats with chat templates
- Sanity checks are critical to catch these issues
