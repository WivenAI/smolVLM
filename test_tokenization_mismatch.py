#!/usr/bin/env python3
"""Test to verify tokenization mismatch bug in label masking"""

from transformers import AutoProcessor
from PIL import Image
import torch

# Load processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", trust_remote_code=True)

# Create test data
question = "What is written in the image?"
answer = "CENTRE"

# Create a dummy image
image = Image.new('RGB', (100, 100), color='white')

# Create messages (same as training code)
user_message = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}]
    }
]

full_messages = user_message + [{"role": "assistant", "content": answer}]

# Get prompt text (question only)
prompt_text = processor.apply_chat_template(user_message, add_generation_prompt=True)

# Get full text (question + answer)
full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False)

print("="*80)
print("PROMPT TEXT (no image):")
print(prompt_text)
print("\n" + "="*80)
print("FULL TEXT (no image):")
print(full_text)
print("\n" + "="*80)

# Tokenize prompt WITHOUT image (like the buggy code does)
prompt_inputs_no_image = processor.tokenizer(prompt_text, return_tensors="pt")
prompt_length_no_image = prompt_inputs_no_image["input_ids"].shape[1]

print(f"\nPrompt tokens (WITHOUT image): {prompt_length_no_image}")
print(f"Token IDs: {prompt_inputs_no_image['input_ids'][0].tolist()}")

# Process full conversation WITH image (like training does)
full_inputs_with_image = processor(
    text=full_text,
    images=image,
    return_tensors="pt",
    padding=True,
    size={"longest_edge": 1024}
)

print(f"\nFull sequence tokens (WITH image): {full_inputs_with_image['input_ids'].shape[1]}")
print(f"Token IDs: {full_inputs_with_image['input_ids'][0].tolist()}")

# Show the BUG: Using prompt_length_no_image to mask full_inputs_with_image
print("\n" + "="*80)
print("THE BUG:")
print(f"Masking first {prompt_length_no_image} tokens...")
print(f"But the full sequence has {full_inputs_with_image['input_ids'].shape[1]} tokens!")
print("\nThis means we're masking the WRONG portion of the sequence!")
print("Image tokens shift everything, so we might be masking part of the answer or missing part of the prompt!")
print("="*80)
