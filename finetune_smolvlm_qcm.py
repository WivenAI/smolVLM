#!/usr/bin/env python3
"""
LoRA Fine-tuning script for SmolVLM-500M-Instruct on QCM dataset
Trains the model to answer multiple choice questions about ERP interface screenshots
Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA
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


class QCMDatasetSFT(torch.utils.data.Dataset):
    """Dataset for SFT training on QCM (multiple choice questions) dataset"""

    def __init__(self, json_path: str, image_dir: str, processor, use_dpo_chosen_only: bool = False):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.use_dpo_chosen_only = use_dpo_chosen_only

        # Load dataset
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Convert DPO format to QCM format if needed
        if use_dpo_chosen_only:
            self.data = self._convert_dpo_to_qcm(raw_data)
            # Keep original DPO items for image access
            self.original_dpo_items = raw_data
            print(f"Loaded {len(self.data)} examples from DPO dataset (chosen responses only)")
        else:
            self.data = raw_data
            self.original_dpo_items = None
            print(f"Loaded {len(self.data)} QCM examples")

    def _convert_dpo_to_qcm(self, dpo_data: List[Dict]) -> List[Dict]:
        """Convert DPO dataset format to QCM format using only chosen responses"""
        qcm_data = []

        for item in dpo_data:
            # Extract the chosen response and prompt
            chosen = item.get('chosen', '')
            prompt = item.get('prompt', '')

            # For DPO dataset, we create a simplified QCM format
            # The "question" contains the prompt and the "correct_answer" is the chosen response
            qcm_item = {
                'question': prompt,
                'options': {'A': chosen},  # Single option with chosen response
                'correct_answer': 'A',
                'explanation': ''
            }
            qcm_data.append(qcm_item)

        return qcm_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image from DPO dataset if available
        if self.use_dpo_chosen_only and self.original_dpo_items:
            dpo_item = self.original_dpo_items[idx]
            image_name = dpo_item.get('image_name', '')
            if image_name:
                image_path = self.image_dir / image_name
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Could not load image {image_path}: {e}")
                    image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            # Create a blank white image for text-only QCM (SmolVLM requires an image)
            image = Image.new('RGB', (224, 224), color='white')

        # Extract QCM data directly from item (no nested 'qcm' key)
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']
        explanation = item.get('explanation', '')

        # Format the prompt with question and options
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"{question}\n\nOptions:\n{options_text}\n\nAnswer:"

        # Format the correct answer with explanation
        correct_option_text = options[correct_answer]
        answer = f"{correct_answer} - {correct_option_text}"
        if explanation:
            answer += f"\n\nExplanation: {explanation}"

        # FIXED: Use chat templates for proper formatting (like proven benchmark approach)
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        full_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        # Apply chat templates
        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        # Process prompt WITH image to get proper prompt length (including image tokens)
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Process full text WITH image
        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # CRITICAL: Mask prompt tokens, only train on answer!
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100  # Mask prompt portion

        # Flatten tensors and add masked labels
        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


def load_model_and_processor(base_model: str = None):
    """Load the SmolVLM model with 4-bit quantization and prepare for LoRA"""
    if base_model is None:
        base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model and processor: {base_model}")

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model with quantization
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
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
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM on QCM dataset")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model to fine-tune (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-500m-qcm-finetuned",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--dataset", type=str, default="dpo_image_dataset/qcm_dataset.json",
                       help="Path to QCM dataset JSON file")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with limited samples")
    parser.add_argument("--use-dpo-chosen-only", action="store_true",
                       help="Use only chosen responses from DPO dataset for SFT training")

    args = parser.parse_args()

    print(f"Starting SmolVLM LoRA fine-tuning on QCM dataset...")
    if args.test:
        print("⚠️  Running in TEST MODE - using only 10 samples")

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name=f"smolvlm-qcm-finetuning{'-test' if args.test else ''}",
        mode="disabled" if args.test else "online"
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor(args.base_model)

    # Create dataset
    print("\nPreparing QCM dataset...")
    full_dataset = QCMDatasetSFT(
        json_path=args.dataset,
        image_dir=args.image_dir,
        processor=processor,
        use_dpo_chosen_only=args.use_dpo_chosen_only
    )

    # In test mode, only use 10 samples
    if args.test:
        indices = list(range(min(10, len(full_dataset))))
        full_dataset = torch.utils.data.Subset(full_dataset, indices)

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
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,  # Must be 1 due to variable image patch sizes
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
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
        gradient_checkpointing=True,
        optim="adamw_8bit",
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

    print("\nStarting LoRA training on QCM dataset...")

    # Train the model
    trainer.train()

    # Save the final model (saves only LoRA adapters)
    print("\nSaving LoRA adapters...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    print(f"\nQCM LoRA Training completed!")
    print(f"LoRA adapters saved to: {args.output_dir}")
    print("\nTo use the fine-tuned model, load the base model and apply the LoRA adapters.")


if __name__ == "__main__":
    main()
