#!/usr/bin/env python3
"""
Inspect samples around position 139 to find the problematic one
"""

import json
from pathlib import Path
from PIL import Image

# Load the dataset
with open('dpo_image_dataset/dpo_dataset_gemini.json', 'r') as f:
    dataset = json.load(f)

print(f"Total samples in dataset: {len(dataset)}")
print("\n" + "="*80)
print("Inspecting samples 135-145 (around where tokenization hangs)")
print("="*80)

# After train_test_split with 300 samples (90/10 split), we get 270 train samples
# The hang happens at sample 139 of 270 train samples
# With the full dataset of 1840 samples, if we select first 300, then split:
# - First 300 samples selected: indices 0-299
# - Train split (90%): 270 samples
# - Sample 139 in train split maps to original index ~154

# Let's check samples 135-145 in the ORIGINAL dataset (before split)
for idx in range(135, 146):
    if idx >= len(dataset):
        break

    sample = dataset[idx]
    print(f"\n{'='*80}")
    print(f"Sample {idx}")
    print(f"{'='*80}")

    # Check if image exists
    image_name = sample.get('image_name', sample.get('image', ''))
    image_path = Path('dpo_image_dataset') / image_name
    image_exists = image_path.exists() if image_name else False

    print(f"Image: {image_name}")
    print(f"Image exists: {image_exists}")

    if image_exists:
        try:
            file_size = image_path.stat().st_size
            print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

            img = Image.open(image_path)
            print(f"Image size: {img.size}")
            print(f"Image mode: {img.mode}")
            print(f"Image format: {img.format}")

            # Check if image is particularly large
            if file_size > 10 * 1024 * 1024:  # > 10MB
                print("⚠️  Very large image file!")
            if img.size[0] * img.size[1] > 10000000:  # > 10 megapixels
                print("⚠️  Very high resolution image!")
        except Exception as e:
            print(f"⚠️  ERROR loading image: {e}")

    # Check text lengths
    prompt = sample.get('prompt', '')
    chosen = sample.get('chosen', '')
    rejected = sample.get('rejected', '')

    print(f"Prompt length: {len(prompt)} chars")
    print(f"Chosen length: {len(chosen)} chars")
    print(f"Rejected length: {len(rejected)} chars")

    # Show first 100 chars of each
    print(f"\nPrompt: {prompt[:100]}...")
    print(f"Chosen: {chosen[:100]}...")
    print(f"Rejected: {rejected[:100]}...")

    # Check for any special characters or anomalies
    if len(prompt) > 5000:
        print("⚠️  Very long prompt!")
    if len(chosen) > 5000:
        print("⚠️  Very long chosen response!")
    if len(rejected) > 5000:
        print("⚠️  Very long rejected response!")

    if not image_exists:
        print("❌ MISSING IMAGE - This could cause the hang!")

print("\n" + "="*80)
print("Inspection complete")
print("="*80)
