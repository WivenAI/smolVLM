#!/usr/bin/env python3
"""
SFT Fine-tuning script for SmolVLM-500M-Instruct
Uses Supervised Fine-Tuning on chosen responses from DPO dataset
More memory-efficient alternative to full DPO training
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
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    default_data_collator
)


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


def load_model_and_processor(use_8bit: bool = True):
    """Load the SmolVLM model and processor

    Args:
        use_8bit: Use 8-bit quantization to reduce memory usage (default: True)
    """
    model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model and processor: {model_name}")
    if use_8bit:
        print("Using 8-bit quantization to reduce memory usage")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load with 8-bit quantization for memory efficiency
    if use_8bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=True
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    return model, processor


def main():
    print("Starting SmolVLM SFT fine-tuning on DPO chosen responses...")

    # Initialize WandB
    wandb.init(project="SmallVLM", name="smolvlm-sft-finetuning")

    # Paths
    dataset_path = "dpo_image_dataset/dpo_dataset_gemini.json"
    image_dir = "dpo_image_dataset"
    output_dir = "./smolvlm-500m-sft-finetuned"

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

    # Training arguments optimized for low memory
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Increased to compensate for batch_size=1
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=2,  # Reduced to save disk space
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),  # Use FP16 if BF16 not supported
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb",
        load_best_model_at_end=False,  # Disabled to save memory
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch",  # Use PyTorch AdamW (more memory efficient)
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # Reduce memory overhead
    )

    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("\nStarting training...")

    # Train the model
    trainer.train()

    # Save the final model
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"\nSFT Training completed!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
