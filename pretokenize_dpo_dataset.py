#!/usr/bin/env python3
"""
Pre-tokenize DPO dataset and save to disk for faster loading
This solves OOM issues by processing in small batches
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoProcessor
from datasets import Dataset


def pretokenize_dpo_dataset(
    json_path: str,
    image_dir: str,
    output_dir: str,
    batch_size: int = 100,
    max_samples: int = None
):
    """
    Pre-tokenize DPO dataset and save to disk in batches

    Args:
        json_path: Path to DPO dataset JSON
        image_dir: Directory containing images
        output_dir: Directory to save tokenized batches
        batch_size: Number of samples to process at once (default: 100)
        max_samples: Maximum number of samples to process (None = all)
    """

    print(f"Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        trust_remote_code=True
    )

    # Load dataset
    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples")
    print(f"Will process in batches of {batch_size}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        'total_samples': len(data),
        'batch_size': batch_size,
        'num_batches': (len(data) + batch_size - 1) // batch_size,
        'source_dataset': json_path,
        'image_dir': image_dir
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessing {metadata['num_batches']} batches...")

    image_dir_path = Path(image_dir)

    # Process in batches
    for batch_idx in range(metadata['num_batches']):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx]

        print(f"\nBatch {batch_idx + 1}/{metadata['num_batches']}: samples {start_idx}-{end_idx-1}")

        batch_dict = {
            'prompt': [],
            'chosen': [],
            'rejected': [],
            'input_ids_chosen': [],
            'attention_mask_chosen': [],
            'pixel_values_chosen': [],
            'input_ids_rejected': [],
            'attention_mask_rejected': [],
            'pixel_values_rejected': [],
        }

        # Process each sample in batch
        for item in tqdm(batch_data, desc=f"Processing batch {batch_idx + 1}"):
            # Load image
            image_name = item.get('image_name', '')
            if image_name:
                image_path = image_dir_path / image_name
                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}, skipping...")
                    continue

                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Resize large images
                max_size = 1536
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            else:
                image = Image.new('RGB', (224, 224), color='white')

            # Tokenize chosen response
            prompt_with_image = f"<image>{item['prompt']}"
            chosen_inputs = processor(
                text=[prompt_with_image + item['chosen']],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Tokenize rejected response
            rejected_inputs = processor(
                text=[prompt_with_image + item['rejected']],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )

            # Store tokenized data
            batch_dict['prompt'].append(item['prompt'])
            batch_dict['chosen'].append(item['chosen'])
            batch_dict['rejected'].append(item['rejected'])

            # Store tensors (squeeze batch dimension)
            batch_dict['input_ids_chosen'].append(chosen_inputs['input_ids'].squeeze(0))
            batch_dict['attention_mask_chosen'].append(chosen_inputs['attention_mask'].squeeze(0))
            batch_dict['pixel_values_chosen'].append(chosen_inputs['pixel_values'].squeeze(0))

            batch_dict['input_ids_rejected'].append(rejected_inputs['input_ids'].squeeze(0))
            batch_dict['attention_mask_rejected'].append(rejected_inputs['attention_mask'].squeeze(0))
            batch_dict['pixel_values_rejected'].append(rejected_inputs['pixel_values'].squeeze(0))

        # Save batch to disk
        batch_file = output_path / f"batch_{batch_idx:04d}.pt"
        print(f"Saving batch to {batch_file}...")
        torch.save(batch_dict, batch_file)

        # Clear memory
        del batch_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✓ Pre-tokenization complete!")
    print(f"  Saved to: {output_path}")
    print(f"  Total batches: {metadata['num_batches']}")
    print(f"\nTo use this tokenized dataset:")
    print(f"  1. Load metadata.json to get batch info")
    print(f"  2. Load batches one at a time with torch.load()")
    print(f"  3. Train incrementally on each batch")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize DPO dataset")
    parser.add_argument("--dataset", type=str,
                       default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str,
                       default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--output-dir", type=str,
                       default="dpo_tokenized",
                       help="Output directory for tokenized data")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for processing (default: 100)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to process (default: all)")

    args = parser.parse_args()

    pretokenize_dpo_dataset(
        json_path=args.dataset,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
