#!/usr/bin/env python3
"""
DPO Fine-tuning using lazy-loaded dataset from disk
Images are loaded on-the-fly, not all at once
"""

import torch
import wandb
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_from_disk


def load_model_and_processor(base_model: str = None):
    if base_model is None:
        base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading base model: {base_model}")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, None, processor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DPO training with lazy-loaded dataset")
    parser.add_argument("--base-model", type=str, default=None,
                       help="Base model")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-dpo-from-disk",
                       help="Output directory")
    parser.add_argument("--dataset-dir", type=str, default="dpo_dataset_lazy",
                       help="Path to saved dataset directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples to use")
    parser.add_argument("--test", action="store_true",
                       help="Test mode with 10 samples")

    args = parser.parse_args()

    print("Starting DPO fine-tuning with lazy-loaded dataset...")
    if args.test:
        print("⚠️  TEST MODE - using 10 samples")

    # Initialize WandB
    wandb.init(
        project="SmallVLM",
        name="smolvlm-dpo-lazy-disk" + ("-test" if args.test else ""),
        mode="disabled" if args.test else "online"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model, ref_model, processor = load_model_and_processor(args.base_model)

    # Load dataset from disk (LAZY LOADING!)
    print(f"\nLoading dataset from {args.dataset_dir}...")
    print("Images will be loaded on-the-fly during training (lazy loading)")

    full_dataset = load_from_disk(args.dataset_dir)
    print(f"✓ Dataset loaded: {len(full_dataset)} samples")

    # Limit samples if requested
    if args.test:
        full_dataset = full_dataset.select(range(min(10, len(full_dataset))))
        print(f"Using {len(full_dataset)} samples (test mode)")
    elif args.max_samples:
        full_dataset = full_dataset.select(range(min(args.max_samples, len(full_dataset))))
        print(f"Using {len(full_dataset)} samples")

    # Split for validation
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")

    # Clear memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # DPO Training config
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
        dataloader_pin_memory=False,
        dataloader_num_workers=0,  # 0 for lazy loading
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
        dataset_num_proc=1,  # Single process for stability
    )

    print("\nInitializing DPO Trainer...")
    print("Note: Tokenization will happen on-the-fly (may be slow initially)")

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
        import traceback
        traceback.print_exc()
        raise

    print("\nStarting DPO training...")

    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    try:
        trainer.train()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA Out of Memory!")
            if torch.cuda.is_available():
                print(f"   Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"\n💡 Try reducing --max-samples")
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
