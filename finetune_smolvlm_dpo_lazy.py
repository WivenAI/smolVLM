#!/usr/bin/env python3
"""
DPO Fine-tuning with LAZY LOADING (memory efficient)
Loads images on-the-fly instead of all at once

PERFORMANCE OPTIMIZATIONS:
1. True lazy image loading - images loaded only when needed during training
2. Pre-tokenization in __getitem__ - tokenize once per sample (not on-the-fly)
3. Optimized collate function - just stacks pre-tokenized tensors
4. Multi-worker DataLoader - parallel data loading/preprocessing
5. Pin memory - faster CPU-to-GPU data transfer
6. Direct PyTorch Dataset - no HF Dataset conversion overhead

Note: DPO training is inherently ~2x slower than SFT because it processes
both chosen AND rejected responses (double forward pass). This is expected.

FIXES:
- Fixed bug where converting to HF Dataset loaded all images into memory
- Tokenization now happens in DataLoader workers (was main thread)
- Uses 2 workers instead of 0 for parallel processing
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
from typing import Dict, List
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


class LazyDPODataset(torch.utils.data.Dataset):
    """
    Lazy-loading DPO dataset that loads images on-the-fly
    Much more memory efficient than loading all at once
    """

    def __init__(self, json_path: str, image_dir: str, processor, max_length: int = 512):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.max_length = max_length

        # Load dataset metadata only
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples (lazy loading mode)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Load and process a single sample on-the-fly with PRE-TOKENIZATION"""
        item = self.data[idx]

        # Load image on-demand
        image_name = item.get('image_name', '')
        if image_name:
            image_path = self.image_dir / image_name
            if not image_path.exists():
                # Fallback to white dummy if image missing
                image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Resize large images
                max_size = 1536
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        else:
            image = Image.new('RGB', (224, 224), color='white')

        # PRE-TOKENIZE for better performance (like SFT does)
        prompt = f"<image>{item['prompt']}"
        chosen = item['chosen']
        rejected = item['rejected']

        # Tokenize chosen response
        chosen_inputs = self.processor(
            text=prompt + chosen,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Tokenize rejected response
        rejected_inputs = self.processor(
            text=prompt + rejected,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        # Return pre-tokenized data (squeeze batch dimension)
        return {
            'input_ids_chosen': chosen_inputs['input_ids'].squeeze(0),
            'attention_mask_chosen': chosen_inputs['attention_mask'].squeeze(0),
            'pixel_values_chosen': chosen_inputs['pixel_values'].squeeze(0),
            'input_ids_rejected': rejected_inputs['input_ids'].squeeze(0),
            'attention_mask_rejected': rejected_inputs['attention_mask'].squeeze(0),
            'pixel_values_rejected': rejected_inputs['pixel_values'].squeeze(0),
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
        }


def collate_fn(batch):
    """
    Custom collate function for pre-tokenized data
    Since tokenization happens in __getitem__, just stack tensors here
    This is much faster than on-the-fly tokenization!
    """
    return {
        'input_ids_chosen': torch.stack([item['input_ids_chosen'] for item in batch]),
        'attention_mask_chosen': torch.stack([item['attention_mask_chosen'] for item in batch]),
        'pixel_values_chosen': torch.stack([item['pixel_values_chosen'] for item in batch]),
        'input_ids_rejected': torch.stack([item['input_ids_rejected'] for item in batch]),
        'attention_mask_rejected': torch.stack([item['attention_mask_rejected'] for item in batch]),
        'pixel_values_rejected': torch.stack([item['pixel_values_rejected'] for item in batch]),
        'prompt': [item['prompt'] for item in batch],
        'chosen': [item['chosen'] for item in batch],
        'rejected': [item['rejected'] for item in batch],
    }


def load_model_and_processor(base_model: str = None):
    if base_model is None:
        base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 4-bit quantization for memory efficiency
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

    # LoRA configuration
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

    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM using DPO with lazy loading")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-500m-dpo-lazy",
                       help="Output directory")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use (default: all)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode with 10 samples")

    args = parser.parse_args()

    print("Starting SmolVLM DPO fine-tuning (LAZY LOADING)...")
    if args.test:
        print("⚠️  Running in TEST MODE - using only 10 samples")

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name="smolvlm-dpo-lazy" + ("-test" if args.test else ""),
        mode="disabled" if args.test else "online"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, ref_model, processor = load_model_and_processor(args.base_model)

    # Create lazy dataset
    print("\nPreparing lazy-loading DPO dataset...")
    full_dataset = LazyDPODataset(
        json_path=args.dataset,
        image_dir=args.image_dir,
        processor=processor,
        max_length=512
    )

    # Limit samples if requested
    if args.test:
        max_samples = min(10, len(full_dataset))
        indices = list(range(max_samples))
        full_dataset.data = [full_dataset.data[i] for i in indices]
    elif args.max_samples:
        max_samples = min(args.max_samples, len(full_dataset))
        full_dataset.data = full_dataset.data[:max_samples]

    print(f"Using {len(full_dataset)} samples")

    # Split PyTorch dataset for validation
    # Don't convert to HF Dataset - use our optimized LazyDPODataset directly!
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    eval_size = dataset_size - train_size

    # Manual split
    import random
    random.seed(42)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    # Create subset datasets
    train_data = [full_dataset.data[i] for i in train_indices]
    eval_data = [full_dataset.data[i] for i in eval_indices]

    # Create separate train and eval datasets
    train_dataset = LazyDPODataset(
        json_path=args.dataset,
        image_dir=args.image_dir,
        processor=processor,
        max_length=512
    )
    train_dataset.data = train_data

    eval_dataset = LazyDPODataset(
        json_path=args.dataset,
        image_dir=args.image_dir,
        processor=processor,
        max_length=512
    )
    eval_dataset.data = eval_data

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Clear memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # DPO Training arguments (OPTIMIZED)
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,  # Enable for faster data transfer
        dataloader_num_workers=2,  # Use 2 workers for parallel data loading (was 0)
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
    )

    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer with lazy loading and pre-tokenization...")
    print("✓ Images are loaded on-the-fly (lazy loading)")
    print("✓ Tokenization happens in DataLoader (pre-tokenization)")
    print("✓ This should be significantly faster than on-the-fly tokenization!")

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
            data_collator=collate_fn,  # Use our optimized pre-tokenized collator
        )
    except Exception as e:
        print(f"\n❌ Error initializing DPO Trainer: {e}")
        import traceback
        traceback.print_exc()
        raise

    print("\nStarting DPO training...")

    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        trainer.train()

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory!")
            print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"\n💡 Try reducing --max-samples or gradient_accumulation_steps")
        import traceback
        traceback.print_exc()
        raise

    # Save model
    print("\nSaving model...")
    try:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"✅ Model saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"\n🎉 DPO Training completed successfully!")


if __name__ == "__main__":
    main()
