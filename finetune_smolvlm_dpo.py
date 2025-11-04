#!/usr/bin/env python3
"""
DPO Fine-tuning script for SmolVLM-500M-Instruct
Uses Direct Preference Optimization (DPO) with the dpo_image_dataset
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List
from dataclasses import dataclass
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


def prepare_dpo_dataset(json_path: str, image_dir: str):
    """Load and prepare DPO dataset in the format expected by DPOTrainer"""

    # Load DPO dataset
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} DPO examples")

    # Prepare dataset in the format DPOTrainer expects
    data_dict = {
        'prompt': [],
        'chosen': [],
        'rejected': [],
        'images': [],
    }

    image_dir_path = Path(image_dir)
    skipped = 0

    for idx, item in enumerate(data):
        try:
            # Load image based on dataset structure
            image_name = item.get('image_name', '')

            if image_name:
                # Image is specified - must exist or crash
                image_path = image_dir_path / image_name

                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Image specified but not found: {image_path}\n"
                        f"Sample index: {idx}\n"
                        f"Dataset: {json_path}\n"
                        f"Check that {image_dir}/ folder contains all referenced images."
                    )

                try:
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    # Resize very large images to avoid OOM
                    max_size = 1536
                    if image.size[0] > max_size or image.size[1] > max_size:
                        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                    if idx % 100 == 0:
                        print(f"✓ Loaded image: {image_path}")

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load image: {image_path}\n"
                        f"Sample index: {idx}\n"
                        f"Error: {e}"
                    )
            else:
                # No image specified - use white dummy (for text-only datasets)
                image = Image.new('RGB', (224, 224), color='white')
                if idx == 0:
                    print("ℹ No image_name in dataset - using white dummy images (text-only mode)")

            # DPOTrainer expects text and will handle tokenization
            # Add <image> token to the prompt for vision models
            data_dict['prompt'].append(f"<image>{item['prompt']}")
            data_dict['chosen'].append(item['chosen'])
            data_dict['rejected'].append(item['rejected'])
            data_dict['images'].append(image)

        except (FileNotFoundError, RuntimeError) as e:
            # Re-raise image loading errors (don't skip them)
            raise
        except Exception as e:
            print(f"Warning: Error processing sample {idx}: {e}, skipping...")
            skipped += 1
            continue

    total_loaded = len(data_dict['prompt'])
    print(f"\n✓ Successfully loaded {total_loaded} DPO samples")
    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} samples due to data errors")
    else:
        print(f"  All samples loaded successfully!")
    return Dataset.from_dict(data_dict)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

def load_model_and_processor(base_model: str = None):
    if base_model is None:
        base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    # Add 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, None, processor

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM using DPO")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model to fine-tune (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-500m-dpo-finetuned",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset_cleaned.json",
                       help="Path to DPO dataset JSON file")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with limited samples")

    args = parser.parse_args()

    print("Starting SmolVLM DPO fine-tuning...")
    if args.test:
        print("⚠️  Running in TEST MODE - using only 10 samples")

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name="smolvlm-dpo-finetuning" + ("-test" if args.test else ""),
        mode="disabled" if args.test else "online"
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, ref_model, processor = load_model_and_processor(args.base_model)

    # Create dataset
    print("\nPreparing DPO dataset...")
    full_dataset = prepare_dpo_dataset(
        json_path=args.dataset,
        image_dir=args.image_dir
    )

    # In test mode, only use 10 samples
    if args.test:
        full_dataset = full_dataset.select(range(min(10, len(full_dataset))))
    else:
        # IMPORTANT: Limit dataset to prevent OOM during tokenization
        # DPOTrainer tokenizes entire dataset during init, which causes OOM with 1840 samples
        # CONFIRMED: 100 samples works (300 hangs at 51%, 100 completes successfully)
        max_samples = 100  # Maximum: 100 samples for 8GB VRAM (tested Oct 31, 2025)
        if len(full_dataset) > max_samples:
            print(f"\n⚠️  Limiting dataset to {max_samples} samples (from {len(full_dataset)}) to prevent OOM")
            print(f"   DPOTrainer tokenizes entire dataset during initialization")
            full_dataset = full_dataset.select(range(max_samples))

    # Split for validation
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Clear memory before DPOTrainer initialization
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # DPO Training arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced from 8 to save memory
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=50,  # Reduced from 100
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,  # Reduced from 100
        save_steps=100,  # Reduced from 200
        save_total_limit=2,  # Reduced from 3
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,  # DPO beta parameter
        loss_type="sigmoid",  # DPO loss type
        max_length=512,  # Limit sequence length to reduce memory
        max_prompt_length=256,  # Limit prompt length
    )

    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer...")
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )
    except Exception as e:
        print(f"\n❌ Error initializing DPO Trainer: {e}")
        print("\nThis usually happens due to:")
        print("  1. OOM during tokenization")
        print("  2. Problematic images in the dataset")
        print("  3. Text sequences that are too long")
        print("\nTry reducing max_length in DPOConfig or using fewer samples.")
        import traceback
        traceback.print_exc()
        raise

    print("\nStarting DPO training...")

    # Monitor GPU memory before training
    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    try:
        # Train the model
        trainer.train()

        # Clear memory after training
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory during training!")
            print(f"   Current memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"   Current memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            print(f"\n💡 Suggestions:")
            print(f"   1. Reduce max_samples further (try 250 or 200)")
            print(f"   2. Reduce gradient_accumulation_steps in DPOConfig")
            print(f"   3. Reduce max_length/max_prompt_length in DPOConfig")
            import traceback
            traceback.print_exc()
            raise
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Save the final model
    print("\nSaving model...")
    try:
        output_dir = args.output_dir
        trainer.save_model(output_dir)
        processor.save_pretrained(output_dir)
        print(f"✅ Model saved successfully to: {output_dir}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")
        print(f"   Training completed but model save failed")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n🎉 DPO Training completed successfully!")
    print(f"Model location: {args.output_dir}")


if __name__ == "__main__":
    main()
