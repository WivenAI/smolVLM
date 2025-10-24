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

    for item in data:
        # Load image
        image_path = image_dir_path / item['image_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # DPOTrainer expects text and will handle tokenization
        # Add <image> token to the prompt for vision models
        data_dict['prompt'].append(f"<image>{item['prompt']}")
        data_dict['chosen'].append(item['chosen'])
        data_dict['rejected'].append(item['rejected'])
        data_dict['images'].append(image)

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
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/dpo_dataset.json",
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

    # Split for validation
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # DPO Training arguments
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,  # DPO beta parameter
        loss_type="sigmoid",  # DPO loss type
    )

    # Initialize DPO Trainer
    print("\nInitializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    print("\nStarting DPO training...")

    # Train the model
    trainer.train()

    # Save the final model
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"\nDPO Training completed!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
