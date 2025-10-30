#!/usr/bin/env python3
"""
Fine-tune SmolVLM on standard benchmarks (DocVQA, OCRBench, etc.)
This serves as a "canary" to verify training works on well-known datasets
before training on custom ERP datasets.

If the model improves on these benchmarks after training, we know:
1. The training code works correctly
2. The model can learn from vision-language data
3. We can proceed with confidence to ERP training
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
from datasets import load_dataset
import argparse


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


class BenchmarkDataset(torch.utils.data.Dataset):
    """Dataset for training on benchmark datasets (DocVQA, OCRBench, etc.)"""

    def __init__(self, benchmark_name: str, split: str, processor, max_samples: int = None):
        self.processor = processor
        self.benchmark_name = benchmark_name
        self.already_limited = False  # Track if we already applied sample limit

        print(f"Loading {benchmark_name} dataset ({split} split)...")

        # Load different benchmarks
        if benchmark_name == "docvqa":
            self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "ocrbench":
            # Try multiple OCR datasets
            try:
                self.dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
            except:
                try:
                    self.dataset = load_dataset("lmms-lab/OCRBench-v2", split="test", trust_remote_code=True)
                except:
                    print("Warning: Could not load OCRBench, using DocVQA instead")
                    self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "textvqa":
            # VQAv2 streaming is very slow (0.1 samples/s) for large sample counts
            # Use streaming only for small counts, otherwise fallback to DocVQA
            sample_limit = max_samples if max_samples else 500

            if sample_limit <= 50:
                # For small sample counts, streaming is acceptable
                try:
                    print(f"Loading VQAv2 with streaming ({sample_limit} samples, ~{sample_limit*10}s)...")
                    from datasets import Dataset as HFDataset
                    dataset_stream = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True, trust_remote_code=True)
                    samples = list(dataset_stream.take(sample_limit))
                    if samples:
                        keys = samples[0].keys()
                        data_dict = {key: [sample[key] for sample in samples] for key in keys}
                        self.dataset = HFDataset.from_dict(data_dict)
                        self.already_limited = True
                        print(f"Successfully loaded {len(self.dataset)} samples via streaming")
                    else:
                        raise ValueError("No samples loaded from stream")
                except Exception as e:
                    print(f"Warning: Streaming failed ({e}), using DocVQA instead")
                    self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
            else:
                # For large sample counts, streaming would take too long (500 samples = ~50 mins)
                # Use DocVQA instead which is optimized and downloads quickly
                print(f"Note: VQAv2 streaming is too slow for {sample_limit} samples (~{sample_limit*10}s)")
                print("Using DocVQA dataset instead (similar VQA task, fast download)...")
                self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Limit samples if specified (skip if already limited during streaming)
        if not self.already_limited and max_samples and len(self.dataset) > max_samples:
            import random
            indices = random.sample(range(len(self.dataset)), max_samples)
            self.dataset = self.dataset.select(indices)

        print(f"Loaded {len(self.dataset)} samples from {benchmark_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract image
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            raise ValueError("No image field found in dataset")

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize large images
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Extract question/answer
        if 'query' in item:
            if isinstance(item['query'], dict):
                question = item['query'].get('en', '')
            else:
                question = item['query']
        elif 'question' in item:
            question = item['question']
        else:
            question = "What do you see in this image?"

        # Extract answer
        if 'answers' in item:
            answers = item['answers']
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0]
            else:
                answer = str(answers)
        elif 'answer' in item:
            answer = item['answer']
        elif 'label' in item:
            answer = str(item['label'])
        else:
            answer = "Unknown"

        # Format as instruction-following
        prompt = f"<image>{question}"
        full_text = f"{prompt}\n{answer}"

        # Process inputs
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Flatten tensors
        for key in inputs:
            inputs[key] = inputs[key].squeeze(0)

        # Set labels for loss computation
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


def load_model_and_processor(base_model: str = None):
    """Load the SmolVLM model with LoRA for efficient training"""
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

    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM on benchmark datasets (canary test)"
    )
    parser.add_argument("--benchmark", type=str,
                       choices=["docvqa", "ocrbench", "textvqa", "chartqa"],
                       default="docvqa",
                       help="Benchmark dataset to train on")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model to fine-tune (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: ./smolvlm-{benchmark}-finetuned)")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Maximum training samples (default: 500)")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with 10 samples")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./smolvlm-{args.benchmark}-finetuned"

    print(f"Starting SmolVLM fine-tuning on {args.benchmark}...")
    if args.test:
        print("⚠️  Running in TEST MODE - using only 10 samples")
        args.max_samples = 10

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name=f"smolvlm-{args.benchmark}-finetuning{'-test' if args.test else ''}",
        mode="disabled" if args.test else "online"
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor(args.base_model)

    # Create dataset
    print("\nPreparing dataset...")
    full_dataset = BenchmarkDataset(
        benchmark_name=args.benchmark,
        split="train",
        processor=processor,
        max_samples=args.max_samples
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
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
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

    # Initialize custom data collator
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

    print(f"\nStarting training on {args.benchmark}...")

    # Train the model
    trainer.train()

    # Save the final model
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    print(f"\n{args.benchmark} Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print(f"\nNext step: Benchmark this model on {args.benchmark} to verify improvement")


if __name__ == "__main__":
    main()
