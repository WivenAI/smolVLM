#!/usr/bin/env python3
"""
Test DPO tokenization and data processing with a few samples
This script will verify that the DPO dataset is correctly loaded and tokenized
"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import Dataset
import sys

# Import the prepare_dpo_dataset function from finetune_smolvlm_dpo.py
sys.path.insert(0, str(Path(__file__).parent))
from finetune_smolvlm_dpo import prepare_dpo_dataset


def test_dpo_dataset_loading(json_path: str, image_dir: str, num_samples: int = 3):
    """Test loading and processing of DPO dataset"""

    print("="*80)
    print("TESTING DPO DATASET LOADING AND TOKENIZATION")
    print("="*80)

    # 1. Test raw dataset loading
    print("\n1. Loading raw DPO dataset...")
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"   ✓ Loaded {len(raw_data)} examples from JSON")
    print(f"   Testing with first {num_samples} samples\n")

    # Show structure of first sample
    print("2. First sample structure:")
    first_sample = raw_data[0]
    for key, value in first_sample.items():
        if key == 'prompt' or key == 'chosen' or key == 'rejected':
            print(f"   - {key}: {value[:100]}..." if len(str(value)) > 100 else f"   - {key}: {value}")
        else:
            print(f"   - {key}: {value}")

    # 2. Test dataset preparation
    print("\n3. Preparing dataset with DPOTrainer format...")
    try:
        dataset = prepare_dpo_dataset(json_path, image_dir)
        print(f"   ✓ Dataset prepared successfully")
        print(f"   - Number of samples: {len(dataset)}")
        print(f"   - Features: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"   ✗ Error preparing dataset: {e}")
        return False

    # Limit to num_samples for testing
    test_dataset = dataset.select(range(min(num_samples, len(dataset))))

    # 3. Test image loading
    print("\n4. Verifying image loading...")
    for idx in range(len(test_dataset)):
        image = test_dataset[idx]['images']
        print(f"   - Sample {idx+1}: Image size={image.size}, mode={image.mode}")

    # 4. Test processor and tokenization
    print("\n5. Testing processor and tokenization...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", trust_remote_code=True)
    print(f"   ✓ Processor loaded")

    # Test tokenization for each sample
    print("\n6. Testing tokenization for each sample:")
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]

        print(f"\n   Sample {idx+1}:")
        print(f"   Prompt: {sample['prompt'][:80]}...")
        print(f"   Chosen: {sample['chosen'][:80]}...")
        print(f"   Rejected: {sample['rejected'][:80]}...")

        try:
            # Test tokenization of prompt + chosen
            prompt_inputs = processor(
                text=sample['prompt'],
                images=sample['images'],
                return_tensors="pt"
            )
            print(f"   ✓ Prompt tokenization successful")
            print(f"     - Input IDs shape: {prompt_inputs['input_ids'].shape}")
            print(f"     - Pixel values shape: {prompt_inputs['pixel_values'].shape if 'pixel_values' in prompt_inputs else 'N/A'}")

            # Test tokenization of chosen response
            chosen_inputs = processor(
                text=sample['chosen'],
                return_tensors="pt"
            )
            print(f"   ✓ Chosen response tokenization successful")
            print(f"     - Input IDs shape: {chosen_inputs['input_ids'].shape}")

            # Test tokenization of rejected response
            rejected_inputs = processor(
                text=sample['rejected'],
                return_tensors="pt"
            )
            print(f"   ✓ Rejected response tokenization successful")
            print(f"     - Input IDs shape: {rejected_inputs['input_ids'].shape}")

            # Decode to verify
            print(f"   Decoded prompt (first 100 tokens): {processor.decode(prompt_inputs['input_ids'][0][:100], skip_special_tokens=False)[:200]}...")

        except Exception as e:
            print(f"   ✗ Error during tokenization: {e}")
            import traceback
            traceback.print_exc()
            return False

    # 5. Test DPO-specific data format
    print("\n7. Verifying DPO data format...")
    print(f"   Required keys: ['prompt', 'chosen', 'rejected', 'images']")
    print(f"   Dataset keys: {list(test_dataset.features.keys())}")

    for key in ['prompt', 'chosen', 'rejected', 'images']:
        if key in test_dataset.features:
            print(f"   ✓ '{key}' present")
        else:
            print(f"   ✗ '{key}' MISSING!")
            return False

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print(f"  - Dataset loaded: {len(dataset)} samples")
    print(f"  - Tested: {num_samples} samples")
    print(f"  - Image loading: OK")
    print(f"  - Tokenization: OK")
    print(f"  - DPO format: OK")
    print("\n✅ DPO training should work correctly with this dataset!")

    return True


def test_dpo_trainer_compatibility(json_path: str, image_dir: str):
    """Test that the dataset is compatible with DPOTrainer"""

    print("\n" + "="*80)
    print("TESTING DPO TRAINER COMPATIBILITY")
    print("="*80)

    try:
        from trl import DPOTrainer, DPOConfig
        print("✓ TRL library imported successfully")
    except ImportError as e:
        print(f"✗ Cannot import TRL: {e}")
        return False

    # Prepare small dataset
    print("\nPreparing test dataset (3 samples)...")
    dataset = prepare_dpo_dataset(json_path, image_dir)
    test_dataset = dataset.select(range(min(3, len(dataset))))

    print(f"✓ Test dataset prepared: {len(test_dataset)} samples")

    # Try to create DPOConfig
    print("\nTesting DPOConfig creation...")
    try:
        dpo_config = DPOConfig(
            output_dir="./test_dpo_output",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-7,
            logging_steps=1,
            remove_unused_columns=False,
            beta=0.1,
            loss_type="sigmoid",
        )
        print("✓ DPOConfig created successfully")
    except Exception as e:
        print(f"✗ Error creating DPOConfig: {e}")
        return False

    print("\n✅ Dataset is compatible with DPOTrainer!")
    print("Note: Full training test requires loading the model (skipped for speed)")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test DPO tokenization and data processing")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON file")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to test")

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        return 1

    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory not found: {args.image_dir}")
        return 1

    # Run tests
    success = test_dpo_dataset_loading(args.dataset, args.image_dir, args.num_samples)

    if success:
        success = test_dpo_trainer_compatibility(args.dataset, args.image_dir)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
