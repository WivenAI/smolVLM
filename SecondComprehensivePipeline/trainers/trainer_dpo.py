"""
DPO Trainer - Direct Preference Optimization for SmolVLM

Features:
- Lazy image loading from disk (images loaded on-the-fly, not all at once)
- Dataset caching to avoid reprocessing
- RAM monitoring logged to WandB
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import gc
import torch
import hashlib
import psutil
from PIL import Image

# Set HuggingFace cache before imports (must be before transformers/peft)
from config.setup import setup_hf_cache, get_hf_cache_dir, BASE_MODEL
setup_hf_cache()

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset, load_from_disk, Features, Value, Sequence, Image as HFImage
import random

# Import dataloader utilities for field extraction (single source of truth)
from dataloader.benchmark_dataset import BenchmarkMixin, BENCHMARK_CONFIGS

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from utils.dual_logger import init_dual_logger, log_metrics
    DUAL_LOGGER_AVAILABLE = True
except ImportError:
    DUAL_LOGGER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ram_usage_gb():
    """Get current RAM usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1e9


def get_system_ram_info():
    """Get system RAM info"""
    mem = psutil.virtual_memory()
    return {
        'total_gb': mem.total / 1e9,
        'available_gb': mem.available / 1e9,
        'used_gb': mem.used / 1e9,
        'percent': mem.percent
    }


# Import shared image utilities
from trainers.image_utils import prepare_image_with_fallback

# Import shared callbacks and model utilities
from trainers.callbacks import DPOEpochEvaluationCallback as EpochEvaluationCallback, RAMMonitorCallback
from trainers.model_utils import load_model_qlora_dpo, resolve_cache_dir

# REMOVED: Local EpochEvaluationCallback - now using shared DPOEpochEvaluationCallback
class DPOTrainerWrapper:
    """Wrapper for DPO training with lazy image loading and dataset caching"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_cache_dir = get_hf_cache_dir()
        # Dataset cache directory
        self.dataset_cache_dir = Path(__file__).parent.parent / "datasets" / "dpo_cache"
        self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, dataset_path: str, dataset_type: str, max_samples: int = None) -> str:
        """Generate a cache key for the dataset"""
        # Create hash from dataset path and config
        key_str = f"{dataset_path}_{dataset_type}_{max_samples}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _get_cached_dataset(self, cache_key: str) -> Optional[Dataset]:
        """Load dataset from cache if it exists"""
        cache_path = self.dataset_cache_dir / cache_key
        if cache_path.exists():
            try:
                logger.info(f"Loading cached dataset from: {cache_path}")
                return load_from_disk(str(cache_path))
            except Exception as e:
                logger.warning(f"Failed to load cached dataset: {e}")
        return None

    def _save_dataset_to_cache(self, dataset: Dataset, cache_key: str):
        """Save dataset to cache"""
        cache_path = self.dataset_cache_dir / cache_key
        try:
            logger.info(f"Saving dataset to cache: {cache_path}")
            dataset.save_to_disk(str(cache_path))
            logger.info(f"Dataset cached successfully")
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")

    def cleanup_cache(self, cache_key: str = None):
        """
        Clean up cached datasets to prevent disk pollution

        Args:
            cache_key: Specific cache to clean, or None to clean all
        """
        import shutil
        freed_space = 0

        if cache_key:
            # Clean specific cache
            cache_path = self.dataset_cache_dir / cache_key
            if cache_path.exists():
                try:
                    size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                    shutil.rmtree(cache_path)
                    freed_space += size
                    logger.info(f"Cleaned cache {cache_key} ({size / 1e9:.2f} GB)")
                except Exception as e:
                    logger.warning(f"Failed to clean cache {cache_key}: {e}")
        else:
            # Clean all caches
            if self.dataset_cache_dir.exists():
                for cache_dir in self.dataset_cache_dir.iterdir():
                    if cache_dir.is_dir():
                        try:
                            size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                            shutil.rmtree(cache_dir)
                            freed_space += size
                            logger.info(f"Cleaned cache {cache_dir.name} ({size / 1e9:.2f} GB)")
                        except Exception as e:
                            logger.warning(f"Failed to clean cache {cache_dir.name}: {e}")

        # Also clean HF datasets cache files
        for pattern in ['cache-*.arrow', '*.lock']:
            for cache_file in self.dataset_cache_dir.glob(f'**/{pattern}'):
                try:
                    size = cache_file.stat().st_size
                    cache_file.unlink()
                    freed_space += size
                except Exception:
                    pass

        if freed_space > 0:
            logger.info(f"Total disk space freed: {freed_space / 1e9:.2f} GB")
        return freed_space

    def load_model(self, base_model: str = None):
        """Load model with QLoRA for DPO training using shared model_utils"""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", BASE_MODEL)

        # Resolve cache directory using shared utility
        cache_dir = resolve_cache_dir(
            self.config.get("model", {}).get("cache_dir", None),
            self.config
        )

        # Use shared model loading function (with do_image_splitting=False for DPO)
        self.model, self.processor = load_model_qlora_dpo(
            base_model=base_model,
            cache_dir=cache_dir
        )

    def prepare_dpo_dataset(self, dataset_path: str, image_dir: str, max_samples: int = None) -> Dataset:
        """Prepare DPO dataset with lazy image loading (images loaded on-the-fly, not all at once)"""
        logger.info(f"Preparing DPO dataset from: {dataset_path}")
        logger.info(f"RAM before dataset prep: {get_ram_usage_gb():.2f} GB")

        # Check cache first
        cache_key = self._get_cache_key(dataset_path, "dpo", max_samples)
        cached_dataset = self._get_cached_dataset(cache_key)
        if cached_dataset is not None:
            logger.info(f"Using cached dataset with {len(cached_dataset)} samples")
            logger.info(f"RAM after loading cache: {get_ram_usage_gb():.2f} GB")
            return cached_dataset

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        image_dir = Path(image_dir).resolve()

        # Build dataset with image PATHS (not loaded images) for lazy loading
        dpo_data = []
        skipped_missing_image = 0
        skipped_no_image_name = 0

        for item in raw_data:
            image_name = item.get('image_name', '')
            image_path_str = None

            if image_name:
                image_path = image_dir / image_name
                if image_path.exists():
                    image_path_str = str(image_path)
                else:
                    skipped_missing_image += 1
                    continue
            else:
                skipped_no_image_name += 1
                continue  # Skip samples without images for now

            prompt = item.get('prompt', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')

            if prompt and chosen and rejected and image_path_str:
                # Store image path instead of loaded image
                dpo_data.append({
                    'prompt_text': prompt,
                    'chosen_text': chosen,
                    'rejected_text': rejected,
                    'image_path': image_path_str,
                })

        # Apply sample limit if specified
        if max_samples is not None and len(dpo_data) > max_samples:
            logger.info(f"Limiting dataset from {len(dpo_data)} to {max_samples} samples")
            dpo_data = dpo_data[:max_samples]

        # Log summary
        total_skipped = skipped_missing_image + skipped_no_image_name
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} samples: {skipped_missing_image} missing images, "
                          f"{skipped_no_image_name} no image_name")

        logger.info(f"Prepared {len(dpo_data)} DPO samples (paths only, images not loaded)")
        logger.info(f"RAM after building paths: {get_ram_usage_gb():.2f} GB")

        # Create dataset with image paths
        dataset = Dataset.from_list(dpo_data)

        # Save to cache
        self._save_dataset_to_cache(dataset, cache_key)

        return dataset

    def _transform_row_to_chat_format(self, row):
        """Transform a single row to chat template format with lazy-loaded image"""
        # Load image on-the-fly
        image_path = row['image_path']
        try:
            image = Image.open(image_path)
            image = prepare_image_with_fallback(image, image_path)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Use placeholder for failed loads
            image = Image.new('RGB', (512, 512), color='black')

        # Build chat format
        return {
            'prompt': [{
                "role": "user",
                "content": [
                    {"type": "image", "text": None},
                    {"type": "text", "text": row['prompt_text']}
                ]
            }],
            'chosen': [{
                "role": "assistant",
                "content": [{"type": "text", "text": row['chosen_text']}]
            }],
            'rejected': [{
                "role": "assistant",
                "content": [{"type": "text", "text": row['rejected_text']}]
            }],
            'images': [image]
        }

    def _apply_chat_transform(self, dataset):
        """Apply chat format transform to dataset (images loaded during map)"""
        logger.info(f"Applying chat transform to {len(dataset)} samples...")
        logger.info(f"RAM before transform: {get_ram_usage_gb():.2f} GB")

        # Remove path columns and add chat format columns
        transformed = dataset.map(
            self._transform_row_to_chat_format,
            remove_columns=['prompt_text', 'chosen_text', 'rejected_text', 'image_path'],
            num_proc=None,  # Disable multiprocessing to avoid CUDA fork error
            desc="Loading images"
        )

        logger.info(f"RAM after transform: {get_ram_usage_gb():.2f} GB")
        return transformed

    def prepare_qcm_dpo_dataset(self, dataset_path: str, image_dir: str, max_samples: int = None) -> Dataset:
        """Prepare DPO dataset from QCM with lazy image loading"""
        logger.info(f"Preparing DPO dataset from QCM: {dataset_path}")
        logger.info(f"RAM before dataset prep: {get_ram_usage_gb():.2f} GB")

        # Check cache first
        cache_key = self._get_cache_key(dataset_path, "qcm_dpo", max_samples)
        cached_dataset = self._get_cached_dataset(cache_key)
        if cached_dataset is not None:
            logger.info(f"Using cached dataset with {len(cached_dataset)} samples")
            logger.info(f"RAM after loading cache: {get_ram_usage_gb():.2f} GB")
            return cached_dataset

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        image_dir = Path(image_dir).resolve()

        # Build dataset with image PATHS for lazy loading
        dpo_data = []
        skipped_missing_image = 0
        skipped_no_image = 0

        for item in raw_data:
            image_name = item.get('image_name', '')
            image_path_str = None

            if image_name:
                image_path = image_dir / image_name
                if image_path.exists():
                    image_path_str = str(image_path)
                else:
                    skipped_missing_image += 1
                    continue
            else:
                skipped_no_image += 1
                continue

            # Get QCM data
            qcm_data = item.get('qcm', item)
            question = qcm_data.get('question', '')
            options = qcm_data.get('options', {})
            correct_answer = qcm_data.get('correct_answer', '')

            if not question or not options or not correct_answer:
                continue

            # Format the question with options
            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter of the correct option:"

            # Chosen = correct answer letter
            chosen = correct_answer

            # Rejected = random wrong answer letter
            wrong_options = [key for key in options.keys() if key != correct_answer]
            rejected = random.choice(wrong_options) if wrong_options else "X"

            # Store paths, not loaded images
            dpo_data.append({
                'prompt_text': prompt,
                'chosen_text': chosen,
                'rejected_text': rejected,
                'image_path': image_path_str,
            })

        # Apply sample limit
        if max_samples is not None and len(dpo_data) > max_samples:
            logger.info(f"Limiting dataset from {len(dpo_data)} to {max_samples} samples")
            dpo_data = dpo_data[:max_samples]

        # Log summary
        total_skipped = skipped_missing_image + skipped_no_image
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} samples: {skipped_missing_image} missing, {skipped_no_image} no image")

        logger.info(f"Prepared {len(dpo_data)} DPO samples from QCM (paths only)")
        logger.info(f"RAM after building paths: {get_ram_usage_gb():.2f} GB")

        dataset = Dataset.from_list(dpo_data)
        self._save_dataset_to_cache(dataset, cache_key)
        return dataset

    def prepare_benchmark_dpo_dataset(self, benchmark_name: str, max_samples: int = None) -> Dataset:
        """Prepare DPO dataset from benchmark by using correct answer as chosen and random wrong answer as rejected"""
        logger.info(f"Preparing DPO dataset from benchmark: {benchmark_name}")

        # Use config from dataloader module (single source of truth)
        if benchmark_name not in BENCHMARK_CONFIGS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_CONFIGS.keys())}")

        config = BENCHMARK_CONFIGS[benchmark_name]
        dataset = load_dataset(config["hf_name"], split=config["split"], trust_remote_code=True)

        question_key = config["question_key"]
        answer_key = config["answer_key"]
        image_key = config["image_key"]

        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        # Collect all answers for generating wrong answers using dataloader extraction
        all_answers = []
        for item in dataset:
            try:
                answer = BenchmarkMixin.extract_answer(item, answer_key)
                all_answers.append(answer)
            except KeyError:
                pass  # Skip items without answers for collection phase

        # Convert to DPO format
        dpo_data = []
        for idx, item in enumerate(dataset):
            # Extract image (will raise KeyError if not found)
            if image_key in item and item[image_key] is not None:
                image = item[image_key]
            elif 'image' in item and item['image'] is not None:
                image = item['image']
            else:
                raise KeyError(f"Sample {idx}: No image found. Tried keys: {image_key}, image. Item keys: {list(item.keys())}")

            # Use fallback chain: no resize → 1920 → 1024 → 512
            image = prepare_image_with_fallback(image, f"benchmark_{benchmark_name}_{idx}")

            # Extract question using dataloader method (will raise KeyError if not found)
            question = BenchmarkMixin.extract_question(item, question_key)

            # Extract correct answer (chosen) using dataloader method (will raise KeyError if not found)
            chosen = BenchmarkMixin.extract_answer(item, answer_key)

            # Generate rejected answer (random wrong answer from other samples)
            rejected_candidates = [a for a in all_answers if a != chosen]
            if not rejected_candidates:
                rejected = "I don't know"
            else:
                rejected = random.choice(rejected_candidates)

            # Format prompt
            prompt = f"Answer briefly. {question}"

            # Use chat template format for TRL VLM DPO
            dpo_data.append({
                'prompt': [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "text": None},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                'chosen': [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": chosen}]
                    }
                ],
                'rejected': [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": rejected}]
                    }
                ],
                'images': [image]
            })

        logger.info(f"Prepared {len(dpo_data)} DPO samples from {benchmark_name}")
        return Dataset.from_list(dpo_data)

    def train_benchmark(self, benchmark_name: str, output_dir: str,
                        use_wandb: bool = True, max_samples: int = None,
                        strategy_name: str = "dpo_benchmark") -> str:
        """Train using DPO on a benchmark dataset"""
        if self.model is None:
            self.load_model()

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        # Initialize dual logger (WandB offline + TensorBoard)
        # Include training type (qlora_dpo) and benchmark in tensorboard path
        tensorboard_dir = f"tensorboard_logs/qlora_dpo_{strategy_name}_{benchmark_name}"
        dual_logger = init_dual_logger(tensorboard_dir, use_wandb=use_wandb and WANDB_AVAILABLE)

        logger.info(f"[QLORA-DPO] Training with DPO on benchmark: {benchmark_name}")
        logger.info(f"[QLORA-DPO] Strategy: {strategy_name}, Benchmark: {benchmark_name}")

        # Prepare dataset
        full_dataset = self.prepare_benchmark_dpo_dataset(benchmark_name, max_samples=max_samples)

        # Split dataset
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Get training config values
        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        # Use DPO-specific learning rate if available, otherwise fall back to general LR
        learning_rate = float(self.config.get("training", {}).get("dpo_learning_rate",
                              self.config.get("training", {}).get("learning_rate", 5e-7)))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 4))

        # DPO config
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            dataset_num_proc=None,  # Disable multiprocessing to avoid CUDA fork error
        )

        # Create callbacks
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        ram_callback = RAMMonitorCallback(log_every_n_steps=10)

        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            callbacks=[eval_callback, ram_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Cleanup memory and cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cleanup_cache()  # Clean up dataset cache to prevent disk pollution

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_qcm(self, dataset_path: str, image_dir: str, output_dir: str,
                  use_wandb: bool = True, max_samples: int = None,
                  strategy_name: str = "dpo_qcm") -> str:
        """Train using DPO on QCM dataset (correct answer as chosen, random wrong as rejected)"""
        if self.model is None:
            self.load_model()

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        # Extract dataset name for logging
        dataset_name = Path(dataset_path).stem

        logger.info(f"[QLORA-DPO] Training with DPO on QCM: {dataset_path}")
        logger.info(f"[QLORA-DPO] Strategy: {strategy_name}, Dataset: {dataset_name}")

        # Prepare dataset
        full_dataset = self.prepare_qcm_dpo_dataset(dataset_path, image_dir, max_samples=max_samples)

        # Split dataset
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Get training config values
        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        learning_rate = float(self.config.get("training", {}).get("dpo_learning_rate",
                              self.config.get("training", {}).get("learning_rate", 5e-7)))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 4))

        # DPO config
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            dataset_num_proc=None,  # Disable multiprocessing to avoid CUDA fork error
        )

        # Apply chat format transform (loads images during map)
        train_dataset = self._apply_chat_transform(train_dataset)
        eval_dataset = self._apply_chat_transform(eval_dataset)

        # Create callbacks
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        ram_callback = RAMMonitorCallback(log_every_n_steps=10)

        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            callbacks=[eval_callback, ram_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Cleanup memory and cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cleanup_cache()  # Clean up dataset cache to prevent disk pollution

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train(self, dataset_path: str, image_dir: str, output_dir: str,
              use_wandb: bool = True, max_samples: int = None,
              strategy_name: str = "dpo") -> str:
        """Train using DPO"""
        if self.model is None:
            self.load_model()

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        # Extract dataset name for logging
        dataset_name = Path(dataset_path).stem

        logger.info(f"[QLORA-DPO] Training with DPO on: {dataset_path}")
        logger.info(f"[QLORA-DPO] Strategy: {strategy_name}, Dataset: {dataset_name}")

        # Prepare dataset
        full_dataset = self.prepare_dpo_dataset(dataset_path, image_dir, max_samples=max_samples)

        # Split dataset
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Get training config values
        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        # Use DPO-specific learning rate if available, otherwise fall back to general LR
        learning_rate = float(self.config.get("training", {}).get("dpo_learning_rate",
                              self.config.get("training", {}).get("learning_rate", 5e-7)))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 4))

        # DPO config
        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            dataset_num_proc=None,  # Disable multiprocessing to avoid CUDA fork error
        )

        # Apply chat format transform (loads images during map)
        train_dataset = self._apply_chat_transform(train_dataset)
        eval_dataset = self._apply_chat_transform(eval_dataset)

        # Create callbacks
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        ram_callback = RAMMonitorCallback(log_every_n_steps=10)

        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,  # Use implicit reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            callbacks=[eval_callback, ram_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        # Cleanup memory and cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.cleanup_cache()  # Clean up dataset cache to prevent disk pollution

        logger.info(f"Model saved to: {output_dir}")
        return output_dir


def train_dpo(config: Dict[str, Any], strategy: Dict[str, Any], output_dir: str,
              base_model: str = None) -> str:
    """
    Train a model using DPO

    Args:
        config: Full configuration
        strategy: Training strategy from config
        output_dir: Where to save the model
        base_model: Base model to start from (can be path to previously trained model)

    Returns:
        Path to trained model
    """
    trainer = DPOTrainerWrapper(config)
    strategy_name = strategy.get("name", "dpo")

    if base_model:
        trainer.load_model(base_model)
    else:
        trainer.load_model()

    base_path = Path(__file__).parent.parent
    dataset_path = base_path / strategy["dataset"]
    image_dir = base_path / strategy["image_dir"]

    return trainer.train(
        dataset_path=str(dataset_path),
        image_dir=str(image_dir),
        output_dir=output_dir,
        use_wandb=config.get("pipeline", {}).get("use_wandb", True),
        max_samples=config.get("training", {}).get("train_samples"),
        strategy_name=strategy_name
    )
