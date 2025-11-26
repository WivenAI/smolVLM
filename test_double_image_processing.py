#!/usr/bin/env python3
"""Test if we're processing images twice in training"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

from transformers import AutoProcessor
from PIL import Image
import torch

# Load processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", trust_remote_code=True)

# Create test data
question = "What is written in the image?"
answer = "CENTRE"
image = Image.new('RGB', (100, 100), color='white')

# Create messages
user_message = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}]
    }
]

full_messages = user_message + [{"role": "assistant", "content": answer}]

# Get prompt text and full text
prompt_text = processor.apply_chat_template(user_message, add_generation_prompt=True)
full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)

print("="*80)
print("PROMPT TEXT:")
print(prompt_text)
print("\n" + "="*80)
print("FULL TEXT:")
print(full_text)
print("\n" + "="*80)

# Process prompt WITH image (what we do now)
prompt_inputs_with_image = processor(
    text=prompt_text,
    images=image,
    return_tensors="pt",
    padding=True,
    size={"longest_edge": 1024}
)

# Process full conversation WITH image (what we always did)
full_inputs_with_image = processor(
    text=full_text,
    images=image,
    return_tensors="pt",
    padding=True,
    size={"longest_edge": 1024}
)

print(f"\nPrompt WITH image: {prompt_inputs_with_image['input_ids'].shape[1]} tokens")
print(f"Full WITH image: {full_inputs_with_image['input_ids'].shape[1]} tokens")
print(f"Difference: {full_inputs_with_image['input_ids'].shape[1] - prompt_inputs_with_image['input_ids'].shape[1]} tokens (should be answer length)")

# Check if prompt text already contains image placeholder
print("\n" + "="*80)
print("CHECKING IF IMAGE PLACEHOLDER IN TEXT:")
print(f"'<image>' in prompt_text: {'<image>' in prompt_text}")
print(f"'<image>' in full_text: {'<image>' in full_text}")

print("\n" + "="*80)
print("TOKEN IDs:")
print(f"Prompt IDs first 20: {prompt_inputs_with_image['input_ids'][0, :20].tolist()}")
print(f"Full IDs first 20: {full_inputs_with_image['input_ids'][0, :20].tolist()}")
print(f"\nLast 20 of prompt: {prompt_inputs_with_image['input_ids'][0, -20:].tolist()}")
print(f"Last 20 of full: {full_inputs_with_image['input_ids'][0, -20:].tolist()}")
