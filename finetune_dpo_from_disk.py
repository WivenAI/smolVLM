#!/usr/bin/env python3
"""
DPO Fine-tuning using lazy-loaded dataset from disk
Images are loaded on-the-fly, not all at once

PERFORMANCE OPTIMIZATIONS:
1. Lazy image loading - images loaded from disk on-demand (not all in memory)
2. Multi-worker data loading - parallel image loading/preprocessing
3. Multi-process dataset operations - faster dataset transformations
4. Pin memory - faster CPU-to-GPU data transfer
5. Automatic cache cleanup - prevents disk space bloat

Note: DPO training is inherently ~2x slower than SFT because it processes
both chosen AND rejected responses for each sample. This is expected behavior.
"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_cache, "datasets")

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


def cleanup_hf_cache(dataset_dir: str = None):
    """
    Clean up HuggingFace datasets cache files

    Args:
        dataset_dir: Dataset directory to clean cache files from
    """
    import shutil
    from pathlib import Path

    freed_space = 0

    # Clean up cache files in dataset directory
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            cache_files = list(dataset_path.glob("cache-*.arrow"))
            if cache_files:
                print(f"Cleaning HF cache files in {dataset_dir}...")
                for cache_file in cache_files:
                    try:
                        size = cache_file.stat().st_size
                        cache_file.unlink()
                        freed_space += size
                        print(f"  ✓ Removed {cache_file.name} ({size / 1e9:.2f} GB)")
                    except Exception as e:
                        print(f"  ⚠ Could not remove {cache_file.name}: {e}")

    # Clean up /tmp HF cache
    tmp_cache = Path("/tmp/hf_datasets_cache")
    if tmp_cache.exists():
        try:
            size = sum(f.stat().st_size for f in tmp_cache.rglob('*') if f.is_file())
            shutil.rmtree(tmp_cache)
            freed_space += size
            print(f"  ✓ Removed /tmp HF cache ({size / 1e9:.2f} GB)")
        except Exception as e:
            print(f"  ⚠ Could not remove /tmp cache: {e}")

    return freed_space


def cleanup_after_training(output_dir: str, dataset_dir: str, cleanup_dataset: bool = False):
    """
    Clean up intermediate files to free disk space after training

    Args:
        output_dir: Training output directory
        dataset_dir: Dataset directory
        cleanup_dataset: Whether to also delete the dataset directory
    """
    import shutil
    from pathlib import Path

    output_path = Path(output_dir)
    freed_space = 0

    # Clean HuggingFace cache files first
    print("Cleaning HuggingFace cache files...")
    freed_space += cleanup_hf_cache(dataset_dir)

    # Remove intermediate checkpoints (keep only final model)
    if output_path.exists():
        print(f"Cleaning checkpoints in {output_dir}...")
        checkpoint_dirs = list(output_path.glob("checkpoint-*"))

        if checkpoint_dirs:
            for checkpoint_dir in checkpoint_dirs:
                try:
                    size = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file())
                    shutil.rmtree(checkpoint_dir)
                    freed_space += size
                    print(f"  ✓ Removed {checkpoint_dir.name} ({size / 1e9:.2f} GB)")
                except Exception as e:
                    print(f"  ⚠ Could not remove {checkpoint_dir.name}: {e}")
        else:
            print("  No checkpoints to remove")

    # Remove runs directory (WandB logs)
    runs_dir = output_path / "runs"
    if runs_dir.exists():
        try:
            size = sum(f.stat().st_size for f in runs_dir.rglob('*') if f.is_file())
            shutil.rmtree(runs_dir)
            freed_space += size
            print(f"  ✓ Removed runs directory ({size / 1e9:.2f} GB)")
        except Exception as e:
            print(f"  ⚠ Could not remove runs directory: {e}")

    # Clean up dataset if requested
    if cleanup_dataset:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            try:
                size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
                shutil.rmtree(dataset_path)
                freed_space += size
                print(f"  ✓ Removed dataset directory {dataset_dir} ({size / 1e9:.2f} GB)")
            except Exception as e:
                print(f"  ⚠ Could not remove dataset: {e}")

    # Clear GPU/CPU cache
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  ✓ Cleared GPU memory cache")

    print(f"\n💾 Total disk space freed: {freed_space / 1e9:.2f} GB")


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
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Keep intermediate checkpoints (cleanup is enabled by default)")
    parser.add_argument("--cleanup-dataset", action="store_true",
                       help="Also delete the dataset directory after training")

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

    # DPO Training config (OPTIMIZED)
    import multiprocessing
    num_workers = min(2, multiprocessing.cpu_count())  # Max 2 workers to avoid overhead

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
        dataloader_pin_memory=True,  # Enable for faster GPU transfer (was False)
        dataloader_num_workers=num_workers,  # Use multiple workers (was 0)
        remove_unused_columns=False,
        report_to="wandb",
        beta=0.1,
        loss_type="sigmoid",
        max_length=512,
        max_prompt_length=256,
        dataset_num_proc=num_workers,  # Use multiple processes for dataset operations (was 1)
    )

    print("\nInitializing DPO Trainer with optimizations...")
    print(f"✓ Using {num_workers} workers for parallel data loading")
    print(f"✓ Using {num_workers} processes for dataset operations")
    print("✓ Pin memory enabled for faster GPU transfer")
    print("✓ Lazy image loading from disk")
    print("Note: Initial tokenization may create cache files in /tmp (cleaned automatically)")

    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor,
        )

        # Clean up cache files created during initialization
        print("\nCleaning up tokenization cache files...")
        freed = cleanup_hf_cache(args.dataset_dir)
        if freed > 0:
            print(f"  ✓ Freed {freed / 1e9:.2f} GB during initialization")

    except Exception as e:
        print(f"\n❌ Error initializing DPO Trainer: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up cache before exiting
        cleanup_hf_cache(args.dataset_dir)
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

    # Cleanup by default (unless --no-cleanup is specified)
    if not args.no_cleanup or args.cleanup_dataset:
        print("\n🧹 Cleaning up...")
        cleanup_after_training(args.output_dir, args.dataset_dir, args.cleanup_dataset)

    print(f"\n🎉 DPO Training completed successfully!")


if __name__ == "__main__":
    main()
