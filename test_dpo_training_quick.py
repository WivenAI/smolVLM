#!/usr/bin/env python3
"""
Quick test of DPO training with minimal samples
This will verify that the full training loop works without running a full training
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoProcessor
from trl import DPOTrainer, DPOConfig
import sys

# Import functions from the main DPO script
sys.path.insert(0, str(Path(__file__).parent))
from finetune_smolvlm_dpo import prepare_dpo_dataset, load_model_and_processor


def test_dpo_training(json_path: str, image_dir: str, num_samples: int = 5):
    """Test DPO training with a few samples"""

    print("="*80)
    print("QUICK DPO TRAINING TEST")
    print("="*80)
    print(f"Testing with {num_samples} samples")
    print(f"This will verify the complete training pipeline\n")

    # 1. Prepare dataset
    print("1. Preparing dataset...")
    try:
        full_dataset = prepare_dpo_dataset(json_path, image_dir)
        print(f"   ✓ Full dataset loaded: {len(full_dataset)} samples")

        # Use only a few samples for quick test
        test_dataset = full_dataset.select(range(min(num_samples, len(full_dataset))))
        print(f"   ✓ Test dataset: {num_samples} samples")

        # Split for train/eval (even with small dataset)
        split = test_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']

        print(f"   - Train samples: {len(train_dataset)}")
        print(f"   - Eval samples: {len(eval_dataset)}")
    except Exception as e:
        print(f"   ✗ Error preparing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. Load model and processor
    print("\n2. Loading model and processor...")
    try:
        model, ref_model, processor = load_model_and_processor()
        print(f"   ✓ Model loaded successfully")
        print(f"   ✓ Processor loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Create DPO config
    print("\n3. Creating DPO training configuration...")
    try:
        training_args = DPOConfig(
            output_dir="./test_dpo_quick",
            num_train_epochs=1,  # Just 1 epoch for testing
            max_steps=3,  # Only 3 steps for quick test
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            warmup_steps=1,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=2,
            save_steps=10,  # Don't save during test
            save_total_limit=1,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",  # Disable reporting for test
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,  # Limit length for speed
            max_prompt_length=256,
        )
        print(f"   ✓ DPO config created")
    except Exception as e:
        print(f"   ✗ Error creating config: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Initialize DPO Trainer
    print("\n4. Initializing DPO Trainer...")
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )
        print(f"   ✓ DPO Trainer initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Test training for a few steps
    print("\n5. Running training test (few steps only)...")
    print("   This may take a few minutes...")
    try:
        # Train for just a few steps (max_steps is in config)
        trainer.train()
        print(f"   ✓ Training completed successfully!")
    except Exception as e:
        print(f"   ✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✅ DPO TRAINING TEST PASSED!")
    print("="*80)
    print("\nSummary:")
    print(f"  - Dataset preparation: OK")
    print(f"  - Model loading: OK")
    print(f"  - DPO Trainer initialization: OK")
    print(f"  - Training execution: OK")
    print("\n✅ Full DPO training pipeline is working correctly!")

    # Clean up test output
    import shutil
    if os.path.exists("./test_dpo_quick"):
        shutil.rmtree("./test_dpo_quick")
        print("\n🧹 Cleaned up test output directory")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quick test of DPO training")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON file")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to test with")

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.dataset):
        print(f"❌ Dataset file not found: {args.dataset}")
        return 1

    if not os.path.exists(args.image_dir):
        print(f"❌ Image directory not found: {args.image_dir}")
        return 1

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠️  CUDA not available, training will be slow\n")

    # Run test
    success = test_dpo_training(args.dataset, args.image_dir, args.num_samples)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
