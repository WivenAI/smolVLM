#!/usr/bin/env python3
"""
Fine-tuning script for HuggingFaceTB/SmolVLM-256M-Instruct
Uses the small Graphcore/vqa dataset for demonstration
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from PIL import Image
import os

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, max_length=512):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Get image and question
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        question = item['question']
        # VQA-RAD dataset has 'answer' field directly
        answer = item['answer'] if 'answer' in item else "unknown"

        # Format with image token
        text = f"<image>Question: {question}\nAnswer: {answer}"

        # Process the inputs
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
    """Load the SmolVLM model and processor"""
    model_name = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model and processor: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    return model, processor

def prepare_datasets():
    """Load and prepare the VQA dataset"""
    print("Loading VQA-RAD dataset (small medical VQA dataset)...")

    # Load the smaller VQA-RAD dataset (2,248 samples total)
    dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

    # Take only a small subset for quick demonstration (50 samples)
    dataset = dataset.select(range(min(50, len(dataset))))

    # Split into train/test
    dataset = dataset.train_test_split(test_size=0.3, seed=42)

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    return dataset['train'], dataset['test']

def main():
    print("Starting SmolVLM fine-tuning...")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor()

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets()

    # Create dataset objects
    train_dataset = VQADataset(train_dataset, processor)
    eval_dataset = VQADataset(eval_dataset, processor)

    print("Setting up training arguments...")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./smolvlm-500m-finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=50,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("Starting training...")

    # Train the model
    trainer.train()

    # Save the final model
    print("Saving model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

    print("Training completed!")
    print(f"Model saved to: {training_args.output_dir}")

if __name__ == "__main__":
    main()