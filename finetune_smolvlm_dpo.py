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
    AutoModelForVision2Seq,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer
from datasets import Dataset


@dataclass
class DPOImageDataset:
    """Custom dataset for DPO training with images"""

    def __init__(self, json_path: str, image_dir: str, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = Path(image_dir)

        # Load DPO dataset
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} DPO examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = self.image_dir / item['image_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        # Format with image token
        prompt_text = f"<image>{prompt}"
        chosen_text = f"<image>{prompt}\n{chosen}"
        rejected_text = f"<image>{prompt}\n{rejected}"

        # Process inputs for chosen response
        chosen_inputs = self.processor(
            text=chosen_text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Process inputs for rejected response
        rejected_inputs = self.processor(
            text=rejected_text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Process prompt only
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Flatten tensors
        result = {
            'prompt_input_ids': prompt_inputs['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_inputs['attention_mask'].squeeze(0),
            'prompt_pixel_values': prompt_inputs['pixel_values'].squeeze(0),
            'chosen_input_ids': chosen_inputs['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_inputs['attention_mask'].squeeze(0),
            'chosen_pixel_values': chosen_inputs['pixel_values'].squeeze(0),
            'rejected_input_ids': rejected_inputs['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_inputs['attention_mask'].squeeze(0),
            'rejected_pixel_values': rejected_inputs['pixel_values'].squeeze(0),
        }

        return result


def prepare_dpo_dataset_for_trainer(dataset_obj):
    """Convert custom dataset to HuggingFace Dataset format"""

    data_dict = {
        'prompt_input_ids': [],
        'prompt_attention_mask': [],
        'prompt_pixel_values': [],
        'chosen_input_ids': [],
        'chosen_attention_mask': [],
        'chosen_pixel_values': [],
        'rejected_input_ids': [],
        'rejected_attention_mask': [],
        'rejected_pixel_values': [],
    }

    for i in range(len(dataset_obj)):
        item = dataset_obj[i]
        for key in data_dict.keys():
            data_dict[key].append(item[key])

    return Dataset.from_dict(data_dict)


def load_model_and_processor():
    """Load the SmolVLM model and processor"""
    model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model and processor: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load model normally (no quantization for DPO training stability)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    # Note: ref_model will be None, DPOTrainer will handle reference internally
    return model, None, processor


def main():
    print("Starting SmolVLM DPO fine-tuning...")

    # Initialize WandB
    wandb.init(project="SmallVLM", name="smolvlm-dpo-finetuning")

    # Paths
    dataset_path = "dpo_image_dataset/dpo_dataset.json"
    image_dir = "dpo_image_dataset"
    output_dir = "./smolvlm-500m-dpo-finetuned"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, ref_model, processor = load_model_and_processor()

    # Create dataset
    print("\nPreparing DPO dataset...")
    dpo_dataset_obj = DPOImageDataset(
        json_path=dataset_path,
        image_dir=image_dir,
        processor=processor,
        max_length=2048
    )

    # Convert to HuggingFace Dataset
    train_dataset = prepare_dpo_dataset_for_trainer(dpo_dataset_obj)

    # Split for validation
    dataset_split = train_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # DPO Training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
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
        tokenizer=processor.tokenizer,
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
