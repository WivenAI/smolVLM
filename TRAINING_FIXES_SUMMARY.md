# Training Fixes Summary - Oct 31, 2025

## Issues Discovered

### Sanity Check Results

**Test 1: Original Format (BROKEN)**
- Base: 53.30%
- After 1 epoch: 16.40% (❌ -36.90%)
- Loss: 16.16 → 0.32 (✅ -98%)

**Test 2: Chat Template (STILL BROKEN)**
- Base: 53.30%
- After 1 epoch: 4.60% (❌ -48.70% WORSE!)
- Loss: 16.45 → 0.32 (✅ -95%)

## Root Causes

### Issue #1: Format Mismatch
**Training format** (lines 196-197):
```python
prompt = f"<image>{question}"
full_text = f"{prompt}\n{answer}"
```

**Evaluation format** (evaluate_ocrbench.py:84-94):
```python
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
```

**Impact:** Model learned one format, evaluated on completely different format.

### Issue #2: Incorrect Label Masking (CRITICAL!)
**Broken code** (line 223):
```python
inputs["labels"] = inputs["input_ids"].clone()
```

**Problem:** Trains on ENTIRE sequence (question + answer)!

**Result:** Model learned to:
- ❌ Regenerate conversation text
- ❌ Respond "The text provided does not contain..."
- ❌ NOT answer based on image content

## Fixes Applied

### Fix #1: Chat Template Format
```python
# Create user message
user_message = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": question}]
}]

# Create full conversation
full_messages = user_message + [{"role": "assistant", "content": answer}]
full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)
```

### Fix #2: Proper Label Masking
```python
# Get prompt text to find its length
prompt_text = processor.apply_chat_template(user_message, add_generation_prompt=True)
prompt_inputs = processor.tokenizer(prompt_text, return_tensors="pt")
prompt_length = prompt_inputs["input_ids"].shape[1]

# Mask prompt tokens - ONLY train on answer!
inputs["labels"] = inputs["input_ids"].clone()
inputs["labels"][:prompt_length] = -100  # Ignore prompt in loss
```

## Verification

✅ Test with 10 samples: Training completes successfully

## Next Steps

**Re-run training with BOTH fixes:**
```bash
python3 finetune_on_benchmarks.py --benchmark ocrbench --num-epochs 1 --output-dir test_ocrbench_FULLY_FIXED
```

**Expected Results:**
- Loss: ↓ (should decrease)
- **Accuracy: ↑ (should INCREASE this time!)**

## Key Lessons

1. **Format consistency is critical:** Training and evaluation MUST use identical formats
2. **Label masking is essential:** Only compute loss on target tokens, not prompts
3. **Sanity checks save time:** Found 2 critical bugs before wasting compute on full training
4. **Loss ≠ Accuracy:** Loss can decrease while accuracy degrades if training is misconfigured
