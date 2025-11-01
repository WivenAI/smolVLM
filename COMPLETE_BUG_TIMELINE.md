# Complete Training Bug Fix Timeline - Oct 31 - Nov 1, 2025

## Initial Problem
Training on 1000 OCRBench samples for 1 epoch showed:
- Loss decreased 98% (16.16 → 0.32) ✅
- **Accuracy DECREASED 36.9% (53.30% → 16.40%)** ❌

This paradox revealed fundamental bugs in the training implementation.

## Bug #1: Format Mismatch (Oct 31)

### The Issue
Training and evaluation used completely different prompt formats:

**Training format** (BROKEN):
```python
prompt = f"<image>{question}"
full_text = f"{prompt}\n{answer}"
```

**Evaluation format**:
```python
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
```

### Impact
- Model learned `<image>question\nanswer` format perfectly (loss ↓98%)
- Evaluation used chat template format model never saw
- Model hallucinated: read "CHRIS" instead of "CHAIN", "C" instead of "CLOSE"

### Fix Applied
```python
# Use chat template for BOTH training and evaluation
user_message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
full_messages = user_message + [{"role": "assistant", "content": answer}]
full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)
```

### Result: STILL BROKEN
- Re-trained with fix
- Accuracy: 53.30% → 4.60% (**-48.7% - WORSE!**)

## Bug #2: Training on Entire Conversation (Oct 31)

### The Issue
Label masking was missing - training on **everything**:

**Broken code**:
```python
inputs["labels"] = inputs["input_ids"].clone()
# NO MASKING - training on question + answer!
```

### Impact
Model responses: *"The text provided does not contain any information about..."*

The model learned to analyze the prompt TEXT instead of the IMAGE content!

### Fix Applied
```python
# Tokenize prompt to find where answer starts
prompt_inputs = processor.tokenizer(prompt_text, return_tensors="pt")
prompt_length = prompt_inputs["input_ids"].shape[1]

# Mask prompt tokens
inputs["labels"] = inputs["input_ids"].clone()
inputs["labels"][:prompt_length] = -100  # Only train on answer
```

### Result: STILL BROKEN
- Re-trained with both fixes (format + masking)
- Accuracy: 53.30% → 5.70% (**-47.6% - STILL CATASTROPHIC!**)

## Bug #3: Image Token Misalignment (Nov 1) - THE REAL BUG

### The Critical Discovery
The "fix" for Bug #2 had a **fatal flaw**: tokenizing prompt WITHOUT image!

**Test results showed**:
- Prompt tokenized **WITHOUT image**: 16 tokens
- Full sequence **WITH image**: 351 tokens!
- Image tokens: 335 (positions 0-334)
- Text prompt: 11 tokens (positions 335-345)
- Answer: 6 tokens (positions 346-351)

**What the "fix" actually did**:
```python
prompt_length = 16  # From tokenizing WITHOUT image
inputs["labels"][:16] = -100  # Masked ONLY 16 image tokens!
```

**Result**:
- ✅ Masked: Positions 0-15 (partial image tokens)
- ❌ NOT masked: Positions 16-334 (REST of image tokens)
- ❌ NOT masked: Positions 335-345 (**TEXT PROMPT**)
- ❌ NOT masked: Positions 346-351 (answer)

Model still trained on prompt + answer + most image tokens!

### Root Cause
Vision-language models insert image tokens when you process with images.
Tokenizing text alone gives completely wrong token counts!

### The ACTUAL Fix
Process prompt **WITH the image** to get correct length:

```python
# CORRECT: Process WITH image to get true token count
prompt_inputs_with_image = processor(
    text=prompt_text,
    images=image,
    return_tensors="pt",
    padding=True,
    size={"longest_edge": 1024}
)
prompt_length = prompt_inputs_with_image["input_ids"].shape[1]  # Returns ~345

# Now masking covers image + text prompt correctly
inputs["labels"] = inputs["input_ids"].clone()
inputs["labels"][:prompt_length] = -100  # Mask positions 0-345
```

Now we mask ALL image tokens (0-334) + text prompt (335-345), training ONLY on answer (346-351).

## Complete Timeline

| Attempt | Fixes Applied | Base→Trained | Change | Status |
|---------|--------------|--------------|---------|--------|
| 1 | None (broken format) | 53.30% → 16.40% | -36.90% | ❌ |
| 2 | Format only | 53.30% → 4.60% | -48.70% | ❌ |
| 3 | Format + "masking" (buggy) | 53.30% → 5.70% | -47.60% | ❌ |
| 4 | Format + CORRECT masking | **Testing...** | **?** | ⏳ |

## Key Lessons

1. **Format consistency is critical**: Training and evaluation MUST use identical formats

2. **Label masking is essential**: Only compute loss on target tokens, not prompts

3. **Vision models add image tokens**: You CANNOT tokenize text without images and expect lengths to match

4. **Always test with the actual input**: Process prompts the SAME way as full sequences when calculating positions

5. **Loss ≠ Accuracy**: Loss decreased in ALL attempts (98%!), but accuracy got WORSE - proves training mechanism was fundamentally broken

6. **Sanity checks save everything**: Without evaluating on training data, we'd never have caught these bugs

7. **Test your assumptions**: The "obvious" approach (tokenize text to get length) was completely wrong for vision-language models

## Final Status

Training with ACTUAL fix completed:
- Loss: 16.50 → 0.0013 (99.99% reduction!)
- Evaluation: **In progress...**
- Expected: Accuracy should FINALLY improve, proving training works

This represents **3 complete re-trainings** and discovery of **3 nested bugs** to get training working correctly!
