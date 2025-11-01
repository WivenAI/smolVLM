# Critical Label Masking Bug - Nov 1, 2025

## The Problem

After applying both format fixes and "proper" label masking, accuracy STILL decreased catastrophically:
- Base: 53.30%
- After "fixed" training: 5.70% (❌ -47.60%)

Model responses showed the SAME issue as before: "The text provided does not contain..."

## Root Cause: Image Tokens Shift Everything!

### The Buggy Code (lines 226-231)
```python
# BUGGY: Tokenize prompt WITHOUT image
prompt_inputs = self.processor.tokenizer(prompt_text, return_tensors="pt")
prompt_length = prompt_inputs["input_ids"].shape[1]  # Returns 16 tokens

inputs["labels"] = inputs["input_ids"].clone()
inputs["labels"][:prompt_length] = -100  # Masks WRONG tokens!
```

### What Actually Happens

When you tokenize text:
- **Without image**: `"What is written in the image?"` = **16 tokens**
- **With image**: Image tokens + text = **351 tokens total!**

Image tokens are inserted at the beginning:
- Positions 0-334: Image tokens (49190, 49189, etc.)
- Positions 335-345: Text prompt "What is written in the image?"
- Positions 346-351: Answer "CENTRE"

### The Bug's Impact

Masking the first 16 tokens means:
- ✅ Masked: Positions 0-15 (partial image tokens)
- ❌ NOT masked: Positions 16-345 (rest of image + **entire text prompt**)
- ❌ NOT masked: Positions 346-351 (answer)

**Result**: Model trained on image tokens, text prompt, AND answer - learning completely wrong patterns!

## The Fix

Process the prompt **WITH the image** to get correct length:

```python
# CORRECT: Process prompt WITH image
prompt_inputs_with_image = self.processor(
    text=prompt_text,
    images=image,
    return_tensors="pt",
    padding=True,
    size={"longest_edge": 1024}
)
prompt_length = prompt_inputs_with_image["input_ids"].shape[1]  # Returns ~345 tokens

inputs["labels"] = inputs["input_ids"].clone()
inputs["labels"][:prompt_length] = -100  # Now masks correctly!
```

Now we mask positions 0-345 (image + text prompt) and only train on 346-351 (answer).

## Timeline of Fixes

1. **Attempt 1** (Oct 31): Format mismatch → 53.30% → 16.40% ❌
2. **Attempt 2** (Oct 31): Fixed format, but training on full conversation → 53.30% → 4.60% ❌
3. **Attempt 3** (Oct 31): "Fixed" label masking (BUGGY - tokenized without image) → 53.30% → 5.70% ❌
4. **Attempt 4** (Nov 1): ACTUAL FIX - process prompt with image → **Testing now...**

## Key Lessons

1. **Vision-language models add image tokens!** You can't tokenize text without images and expect lengths to match.
2. **Always process prompts the same way as full sequences** when calculating masking positions.
3. **Decreasing loss ≠ working training** - Loss decreased in all attempts, but model learned wrong patterns.
4. **Test your assumptions** - The "obvious" tokenization approach was completely wrong.
5. **Sanity checks are essential** - Without evaluating on training data, we'd never have caught this.
