#!/usr/bin/env python3
"""
Check how many samples have missing images in the dataset
"""

import json
from pathlib import Path

# Load the dataset
with open('dpo_image_dataset/dpo_dataset_gemini.json', 'r') as f:
    dataset = json.load(f)

print(f"Total samples: {len(dataset)}")

missing_images = []
valid_samples = []

for idx, sample in enumerate(dataset):
    image_name = sample.get('image_name', sample.get('image', ''))
    image_path = Path('dpo_image_dataset') / image_name

    if not image_name or image_name == 'N/A' or not image_path.exists():
        missing_images.append(idx)
    else:
        valid_samples.append(sample)

print(f"\n{'='*80}")
print(f"Samples with missing images: {len(missing_images)}")
print(f"Valid samples with images: {len(valid_samples)}")
print(f"{'='*80}")

print(f"\nFirst 20 indices with missing images: {missing_images[:20]}")
print(f"Last 20 indices with missing images: {missing_images[-20:]}")

# Check if samples 135-145 are in the missing list
print(f"\nSamples 135-145 missing? {all(i in missing_images for i in range(135, 146))}")

# Create a cleaned dataset
cleaned_dataset_path = 'dpo_image_dataset/dpo_dataset_gemini.json'
with open(cleaned_dataset_path, 'w') as f:
    json.dump(valid_samples, f, indent=2, ensure_ascii=False)

print(f"\n✅ Created cleaned dataset: {cleaned_dataset_path}")
print(f"   Total valid samples: {len(valid_samples)}")
