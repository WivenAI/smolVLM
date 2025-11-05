#!/usr/bin/env python3
"""
Test script to verify image loading in QCM and DPO datasets
"""

import json
import sys
from pathlib import Path
from PIL import Image

def test_qcm_dataset(json_path, image_dir):
    """Test QCM dataset image loading"""
    print(f"\n{'='*80}")
    print(f"Testing QCM Dataset: {json_path}")
    print(f"{'='*80}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Check structure
    if data and 'qcm' in data[0]:
        print("Structure: Nested (image_name at top level, qcm nested)")
        has_images = sum(1 for item in data if item.get('image_name'))
        print(f"Samples with image_name: {has_images}")
    else:
        print("Structure: Flat (simple QCM format)")
        has_images = sum(1 for item in data if item.get('image_name'))
        print(f"Samples with image_name: {has_images}")

    # Test loading first 5 images
    image_dir_path = Path(image_dir)
    loaded = 0
    missing = 0
    no_image_name = 0

    for i, item in enumerate(data[:min(5, len(data))]):
        image_name = item.get('image_name', '')
        if image_name:
            image_path = image_dir_path / image_name
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    print(f"  ✓ Sample {i}: Loaded {image_name} ({img.size})")
                    loaded += 1
                except Exception as e:
                    print(f"  ✗ Sample {i}: Error loading {image_name}: {e}")
                    missing += 1
            else:
                print(f"  ✗ Sample {i}: Image not found: {image_path}")
                missing += 1
        else:
            print(f"  ○ Sample {i}: No image_name (will use white dummy)")
            no_image_name += 1

    print(f"\nSummary: {loaded} loaded, {missing} missing, {no_image_name} no image_name")
    return loaded > 0 or no_image_name > 0


def test_dpo_dataset(json_path, image_dir):
    """Test DPO dataset image loading"""
    print(f"\n{'='*80}")
    print(f"Testing DPO Dataset: {json_path}")
    print(f"{'='*80}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    # Check for images
    has_images = sum(1 for item in data if item.get('image_name'))
    print(f"Samples with image_name: {has_images}")

    # Test loading first 5 images
    image_dir_path = Path(image_dir)
    loaded = 0
    missing = 0
    no_image_name = 0

    for i, item in enumerate(data[:min(5, len(data))]):
        image_name = item.get('image_name', '')
        if image_name:
            image_path = image_dir_path / image_name
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    print(f"  ✓ Sample {i}: Loaded {image_name} ({img.size})")
                    loaded += 1
                except Exception as e:
                    print(f"  ✗ Sample {i}: Error loading {image_name}: {e}")
                    missing += 1
            else:
                print(f"  ✗ Sample {i}: Image not found: {image_path}")
                missing += 1
        else:
            print(f"  ○ Sample {i}: No image_name (will use white dummy)")
            no_image_name += 1

    print(f"\nSummary: {loaded} loaded, {missing} missing, {no_image_name} no image_name")
    return loaded > 0 or no_image_name > 0


def main():
    print("\n" + "="*80)
    print("IMAGE LOADING VERIFICATION TEST")
    print("="*80)

    image_dir = "dpo_image_dataset"

    # Test all datasets
    datasets = [
        ("QCM (nested with images)", "dpo_image_dataset/qcm/qcm_dataset_gemini.json", image_dir),
        ("QCM Nova Pro", "dpo_image_dataset/qcm/qcm_dataset_nova_pro.json", image_dir),
        ("DPO Gemini", "dpo_image_dataset/dpo_dataset_gemini.json", image_dir),
        ("DPO Nova Pro", "dpo_image_dataset/dpo_dataset_nova_pro.json", image_dir),
        ("Balanced QCM (text-only)", "balanced_qcm_all_end.json", image_dir),
    ]

    results = {}
    for name, json_path, img_dir in datasets:
        if Path(json_path).exists():
            if "dpo" in name.lower() and "balanced" not in name.lower():
                results[name] = test_dpo_dataset(json_path, img_dir)
            else:
                results[name] = test_qcm_dataset(json_path, img_dir)
        else:
            print(f"\n⚠️  Skipping {name}: File not found at {json_path}")
            results[name] = False

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_pass = all(results.values())
    print("\n" + ("="*80))
    if all_pass:
        print("✓ ALL TESTS PASSED - Image loading working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Check output above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
