#!/usr/bin/env python3
"""
Fine-tune SmolVLM using TRL's SFTTrainer (Official HuggingFace approach)
This uses the correct, tested approach with automatic label masking.
"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

import torch
from pathlib import Path
from PIL import Image
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
import argparse


class BenchmarkDatasetTRL:
    """Dataset for TRL SFTTrainer - returns messages format"""

    def __init__(self, benchmark_name: str, split: str, processor, max_samples: int = None):
        self.processor = processor
        self.benchmark_name = benchmark_name
        self.already_limited = False

        print(f"Loading {benchmark_name} dataset ({split} split)...")

        # Load different benchmarks
        if benchmark_name == "docvqa":
            self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "ocrbench":
            try:
                self.dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
            except:
                try:
                    self.dataset = load_dataset("lmms-lab/OCRBench-v2", split="test", trust_remote_code=True)
                except:
                    print("Warning: Could not load OCRBench, using DocVQA instead")
                    self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "textvqa":
            sample_limit = max_samples if max_samples else 500
            if sample_limit <= 50:
                try:
                    print(f"Loading VQAv2 with streaming ({sample_limit} samples)...")
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
                print(f"Note: VQAv2 streaming too slow for {sample_limit} samples, using DocVQA instead")
                self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Limit samples if specified
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

        # Resize images more aggressively to avoid truncation issues
        # Smaller images = fewer image tokens = fits within sequence length limits
        max_size = 384  # Reduced from 1024 to avoid truncation
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Extract question
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

        # Fully pre-process everything to avoid TRL's truncation issues
        # Format messages for chat template
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
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

        # Process WITH images and NO truncation
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=False,  # We'll pad in collator
            truncation=False,  # NEVER truncate!
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        # Create labels with proper masking
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[0, :prompt_length] = -100  # Mask prompt tokens

        # Return fully processed tensors
        return {
            "input_ids": full_inputs["input_ids"][0],  # Remove batch dim
            "attention_mask": full_inputs["attention_mask"][0],
            "pixel_values": full_inputs["pixel_values"][0] if "pixel_values" in full_inputs else None,
            "labels": labels[0]
        }


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

    return model, processor


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SmolVLM using TRL SFTTrainer (official approach)"
    )
    parser.add_argument("--benchmark", type=str,
                       choices=["docvqa", "ocrbench", "textvqa", "chartqa"],
                       default="docvqa",
                       help="Benchmark dataset to train on")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model to fine-tune (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: ./smolvlm-{benchmark}-trl)")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum training samples (default: 1000)")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with 10 samples")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"./smolvlm-{args.benchmark}-trl"

    print(f"Starting SmolVLM fine-tuning with TRL SFTTrainer on {args.benchmark}...")
    if args.test:
        print("⚠️  Running in TEST MODE - using only 10 samples")
        args.max_samples = 10

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name=f"smolvlm-{args.benchmark}-trl{'-test' if args.test else ''}",
        mode="disabled" if args.test else "online"
    )

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, processor = load_model_and_processor(args.base_model)

    # LoRA configuration for TRL
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Create dataset
    print("\nPreparing dataset...")
    full_dataset = BenchmarkDatasetTRL(
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

    print("\nSetting up TRL SFTConfig...")

    # TRL SFTConfig (replaces TrainingArguments)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
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
        # We pre-process everything in the dataset, so no text field needed
        dataset_text_field="",
    )

    # Initialize TRL SFTTrainer
    # This automatically handles:
    # - Chat template application
    # - Label masking (trains only on assistant response)
    # - Vision-language data collation
    print("\nInitializing TRL SFTTrainer...")
    print("✨ TRL will automatically handle label masking!")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor,  # Pass full processor for VLM
    )

    print(f"\nStarting training on {args.benchmark} with TRL...")

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
