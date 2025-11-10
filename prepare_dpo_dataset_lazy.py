#!/usr/bin/env python3
"""
Prepare DPO dataset for lazy loading with HuggingFace Datasets
Images are stored as paths and decoded on-the-fly during training
"""

import json
from pathlib import Path
from datasets import Dataset, Features, Image, Value


def prepare_lazy_dpo_dataset(
    json_path: str,
    image_dir: str,
    output_dir: str,
    max_samples: int = None
):
    """
    Create HuggingFace Dataset with lazy image loading

    Images are stored as file paths, not loaded into memory.
    They will be decoded on-the-fly during training.

    Args:
        json_path: Path to DPO dataset JSON
        image_dir: Directory containing images
        output_dir: Directory to save the dataset
        max_samples: Maximum samples to include (None = all)
    """

    print(f"Loading dataset from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples")

    image_dir_path = Path(image_dir)

    # Prepare data with image PATHS only (not loaded images)
    dataset_dict = {
        'prompt': [],
        'chosen': [],
        'rejected': [],
        'image_path': [],  # Store paths, not images!
    }

    skipped = 0
    for idx, item in enumerate(data):
        image_name = item.get('image_name', '')

        if image_name:
            image_path = str(image_dir_path / image_name)

            # Verify image exists
            if not Path(image_path).exists():
                print(f"Warning: Image not found: {image_path}, skipping...")
                skipped += 1
                continue
        else:
            # For text-only, we'll handle this later
            image_path = None

        dataset_dict['prompt'].append(f"<image>{item['prompt']}")
        dataset_dict['chosen'].append(item['chosen'])
        dataset_dict['rejected'].append(item['rejected'])
        dataset_dict['image_path'].append(image_path)

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{len(data)} samples...")

    print(f"\n✓ Prepared {len(dataset_dict['prompt'])} samples")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} samples (missing images)")

    # Define features with Image type for lazy loading
    features = Features({
        'prompt': Value('string'),
        'chosen': Value('string'),
        'rejected': Value('string'),
        'image_path': Value('string'),  # Will convert to Image later
    })

    # Create HuggingFace Dataset
    print("\nCreating HuggingFace Dataset...")
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Cast image_path column to Image type for lazy loading
    # This tells HF Datasets to decode images on-the-fly
    print("Setting up lazy image loading...")
    dataset = dataset.cast_column('image_path', Image())

    # Rename column to 'images' for DPOTrainer compatibility
    dataset = dataset.rename_column('image_path', 'images')

    # Save to disk
    output_path = Path(output_dir)
    print(f"\nSaving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))

    print(f"\n✓ Dataset saved successfully!")
    print(f"  Location: {output_path}")
    print(f"  Total samples: {len(dataset)}")
    print(f"\nTo use this dataset:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{output_path}')")
    print(f"  # Images will be loaded lazily during training!")

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare lazy-loading DPO dataset")
    parser.add_argument("--dataset", type=str,
                       default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str,
                       default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--output-dir", type=str,
                       default="dpo_dataset_lazy",
                       help="Output directory for dataset")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples (default: all)")

    args = parser.parse_args()

    prepare_lazy_dpo_dataset(
        json_path=args.dataset,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
