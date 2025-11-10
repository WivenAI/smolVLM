#!/usr/bin/env python3
"""
DPO Fine-tuning script with BATCHED processing
Processes dataset in batches of 100 samples to avoid OOM
Goes through the entire dataset incrementally
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


def prepare_dpo_batch(data_batch: list, image_dir: str):
    """Load and prepare a single batch of DPO data"""

    data_dict = {
        'prompt': [],
        'chosen': [],
        'rejected': [],
        'images': [],
    }

    image_dir_path = Path(image_dir)
    skipped = 0

    for idx, item in enumerate(data_batch):
        try:
            # Load image
            image_name = item.get('image_name', '')

            if image_name:
                image_path = image_dir_path / image_name

                if not image_path.exists():
                    print(f"Warning: Image not found: {image_path}, skipping...")
                    skipped += 1
                    continue

                try:
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Resize very large images
                    max_size = 1536
                    if image.size[0] > max_size or image.size[1] > max_size:
                        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                except Exception as e:
                    print(f"Warning: Failed to load image {image_path}: {e}, skipping...")
                    skipped += 1
                    continue
            else:
                # No image - use white dummy
                image = Image.new('RGB', (224, 224), color='white')

            # Add to batch
            data_dict['prompt'].append(f"<image>{item['prompt']}")
            data_dict['chosen'].append(item['chosen'])
            data_dict['rejected'].append(item['rejected'])
            data_dict['images'].append(image)

        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}, skipping...")
            skipped += 1
            continue

    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} samples in this batch")

    return Dataset.from_dict(data_dict)


def load_model_and_processor(base_model: str = None):
    if base_model is None:
        base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, None, processor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM using DPO in batches")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-500m-dpo-batched",
                       help="Output directory")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of samples per batch (default: 100)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode with 2 batches of 10 samples each")

    args = parser.parse_args()

    print("Starting SmolVLM DPO fine-tuning with BATCHED processing...")
    if args.test:
        print("⚠️  TEST MODE - using 2 batches of 10 samples")
        batch_size = 10
    else:
        batch_size = args.batch_size

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name="smolvlm-dpo-batched" + ("-test" if args.test else ""),
        mode="disabled" if args.test else "online"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, ref_model, processor = load_model_and_processor(args.base_model)

    # Load dataset metadata only (no images yet!)
    print(f"\nLoading dataset metadata from {args.dataset}...")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    total_samples = len(all_data)
    print(f"✓ Total samples: {total_samples}")

    # Split into train/eval BEFORE batching
    from random import shuffle, seed as random_seed
    random_seed(42)

    indices = list(range(total_samples))
    shuffle(indices)

    eval_size = int(0.1 * total_samples)
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]

    train_data = [all_data[i] for i in train_indices]
    eval_data = [all_data[i] for i in eval_indices]

    print(f"Train samples: {len(train_data)}")
    print(f"Eval samples: {len(eval_data)}")

    # In test mode, limit to 2 batches
    if args.test:
        train_data = train_data[:20]  # 2 batches of 10
        eval_data = eval_data[:10]
        print(f"Test mode: Limited to {len(train_data)} train, {len(eval_data)} eval")

    # Calculate number of batches
    num_batches = (len(train_data) + batch_size - 1) // batch_size
    print(f"\n📦 Processing {num_batches} batches of ~{batch_size} samples each")

    # Prepare eval dataset once (small enough to keep in memory)
    print(f"\n🔄 Preparing evaluation dataset ({len(eval_data)} samples)...")
    eval_dataset = prepare_dpo_batch(eval_data, args.image_dir)
    print(f"✓ Eval dataset ready: {len(eval_dataset)} samples")

    # DPO Training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,  # 1 epoch per batch, we'll do 3 passes total
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=10,  # Fewer warmup steps per batch
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=1000000,  # Don't save during batch training
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
    )

    # Process batches
    import gc
    total_epochs = 3  # Go through dataset 3 times total

    for epoch in range(total_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{total_epochs}")
        print(f"{'='*60}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_data = train_data[start_idx:end_idx]

            print(f"\n{'─'*60}")
            print(f"Batch {batch_idx + 1}/{num_batches} (Epoch {epoch + 1})")
            print(f"Samples {start_idx}-{end_idx-1} ({len(batch_data)} samples)")
            print(f"{'─'*60}")

            # Prepare batch dataset (loads images for this batch only)
            print(f"Loading batch images...")
            train_batch_dataset = prepare_dpo_batch(batch_data, args.image_dir)
            print(f"✓ Batch dataset ready: {len(train_batch_dataset)} samples")

            # Clear memory before training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize trainer for this batch
            print(f"Initializing DPO Trainer for batch {batch_idx + 1}...")
            try:
                trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    args=training_args,
                    train_dataset=train_batch_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=processor,
                )
            except Exception as e:
                print(f"\n❌ Error initializing trainer: {e}")
                import traceback
                traceback.print_exc()
                continue

            print(f"Training on batch {batch_idx + 1}...")

            try:
                # Train on this batch
                trainer.train()
                print(f"✓ Batch {batch_idx + 1} completed successfully")

                # Clear memory after batch
                del trainer
                del train_batch_dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n❌ OOM on batch {batch_idx + 1}!")
                    if torch.cuda.is_available():
                        print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"   Skipping this batch and continuing...")

                    # Clear memory and continue
                    del trainer
                    del train_batch_dataset
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
            except Exception as e:
                print(f"\n❌ Error training batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n✓ Epoch {epoch + 1} completed")

    # Save final model
    print("\n" + "="*60)
    print("Saving final model...")
    print("="*60)
    try:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"✅ Model saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n🎉 DPO Training completed successfully!")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Batches per epoch: {num_batches}")
    print(f"   Model location: {args.output_dir}")


if __name__ == "__main__":
    main()
