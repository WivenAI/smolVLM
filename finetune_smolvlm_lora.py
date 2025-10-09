#!/usr/bin/env python3
"""
LoRA Fine-tuning script for SmolVLM-500M-Instruct
Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA on chosen responses
Memory-efficient approach that works on consumer GPUs
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VisionLanguageDataCollator:
    """Custom data collator for vision-language models"""

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate pixel_values from text features
        pixel_values = [f.pop('pixel_values') for f in features]

        # Find max length for padding
        max_length = max(f['input_ids'].shape[0] for f in features)

        # Pad text features
        batch = {}
        batch['pixel_values'] = torch.stack(pixel_values)

        # Pad input_ids, attention_mask, and labels
        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = f['input_ids'].shape[0]
            pad_len = max_length - seq_len

            # Pad input_ids
            input_ids.append(torch.cat([
                f['input_ids'],
                torch.full((pad_len,), 0, dtype=f['input_ids'].dtype)
            ]))

            # Pad attention_mask
            attention_mask.append(torch.cat([
                f['attention_mask'],
                torch.zeros(pad_len, dtype=f['attention_mask'].dtype)
            ]))

            # Pad labels (use -100 for padding to ignore in loss)
            labels.append(torch.cat([
                f['labels'],
                torch.full((pad_len,), -100, dtype=f['labels'].dtype)
            ]))

        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['labels'] = torch.stack(labels)

        return batch


class DPOImageDatasetSFT(torch.utils.data.Dataset):
    """Dataset for SFT training using chosen responses from DPO dataset"""

    def __init__(self, json_path: str, image_dir: str, processor):
        self.processor = processor
        self.image_dir = Path(image_dir)

        # Load DPO dataset
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = self.image_dir / item['image_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Use prompt and chosen response only
        prompt = item['prompt']
        chosen = item['chosen']

        # Format: <image>prompt\nchosen_response
        text = f"<image>{prompt}\n{chosen}"

        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Flatten tensors
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)

        # Set labels for loss computation
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


def load_model_and_processor():
    """Load the SmolVLM model with 4-bit quantization and prepare for LoRA"""
    model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model and processor: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration - target language model layers
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def main():
    print("Starting SmolVLM LoRA fine-tuning on DPO chosen responses...")

    # Initialize WandB
    wandb.init(project="SmallVLM", name="smolvlm-lora-finetuning")

    # Paths
    dataset_path = "dpo_image_dataset/dpo_dataset.json"
    image_dir = "dpo_image_dataset"
    output_dir = "./smolvlm-500m-lora-finetuned"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor()

    # Create dataset
    print("\nPreparing dataset...")
    full_dataset = DPOImageDatasetSFT(
        json_path=dataset_path,
        image_dir=image_dir,
        processor=processor
    )

    # Split into train/eval
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    print("\nSetting up training arguments...")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Must be 1 due to variable image patch sizes
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,  # Low LR (1e-5 / 10 = 1e-6) for stability
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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_8bit",  # Use 8-bit optimizer for memory efficiency
    )

    # Initialize custom data collator for vision-language models
    data_collator = VisionLanguageDataCollator()

    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("\nStarting LoRA training...")

    # Train the model
    trainer.train()

    # Save the final model (saves only LoRA adapters)
    print("\nSaving LoRA adapters...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"\nLoRA Training completed!")
    print(f"LoRA adapters saved to: {output_dir}")
    print("\nTo use the fine-tuned model, load the base model and apply the LoRA adapters.")


if __name__ == "__main__":
    main()
