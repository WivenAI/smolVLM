"""
SFT Trainer - Supervised Fine-Tuning for SmolVLM
Handles both benchmark training and QCM training
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import torch
from PIL import Image
from dataclasses import dataclass

# Set HuggingFace cache before imports (must be before transformers/peft)
from config.setup import setup_hf_cache, get_hf_cache_dir, BASE_MODEL
setup_hf_cache()

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Import dataloader utilities for field extraction (single source of truth)
from dataloader.benchmark_dataset import BenchmarkMixin, BENCHMARK_CONFIGS
from dataloader.data_collators import VisionLanguageDataCollator

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

# Import shared image utilities
from trainers.image_utils import prepare_image_with_fallback

# Import shared callbacks and model utilities
from trainers.callbacks import SFTEpochEvaluationCallback as EpochEvaluationCallback
from trainers.model_utils import load_model_qlora, resolve_cache_dir


class QCMDataset(torch.utils.data.Dataset):
    """Dataset for QCM (multiple choice questions) training"""

    def __init__(self, json_path: str, image_dir: str, processor):
        self.processor = processor
        self.image_dir = Path(image_dir)

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.original_items = raw_data
        if raw_data and 'qcm' in raw_data[0]:
            self.data = [item['qcm'] for item in raw_data]
        else:
            self.data = raw_data

        logger.info(f"Loaded {len(self.data)} QCM examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        original_item = self.original_items[idx]

        # Load image
        image_name = original_item.get('image_name', '')
        if image_name:
            image_path = self.image_dir / image_name
            if image_path.exists():
                image = Image.open(image_path)
                image = prepare_image_with_fallback(image, str(image_path))
            else:
                image = Image.new('RGB', (512, 512), color='white')
        else:
            image = Image.new('RGB', (512, 512), color='white')

        # Format prompt
        qcm_data = item.get('qcm', item)
        question = qcm_data['question']
        options = qcm_data['options']
        correct_answer = qcm_data['correct_answer']

        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter of the correct option:"

        # Train to output just the letter (matching evaluation format)
        answer = correct_answer

        # Create messages
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

        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Mask prompt tokens
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


class DPOSFTDataset(torch.utils.data.Dataset):
    """Dataset for SFT training on DPO dataset (using chosen responses)"""

    def __init__(self, json_path: str, image_dir: str, processor):
        self.processor = processor
        self.image_dir = Path(image_dir)

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} DPO examples for SFT")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_name = item.get('image_name', '')
        if image_name:
            image_path = self.image_dir / image_name
            if image_path.exists():
                image = Image.open(image_path)
                image = prepare_image_with_fallback(image, str(image_path))
            else:
                image = Image.new('RGB', (512, 512), color='white')
        else:
            image = Image.new('RGB', (512, 512), color='white')

        prompt = item['prompt']
        chosen_response = item['chosen']  # Use the good response for SFT

        # Create messages
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
                "content": [{"type": "text", "text": chosen_response}]
            }
        ]

        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Mask prompt tokens (only train on response)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


class BenchmarkDataset(torch.utils.data.Dataset):
    """Dataset for training on benchmark datasets (DocVQA, OCRBench, ChartQA)"""

    def __init__(self, benchmark_name: str, processor, max_samples: int = None):
        self.processor = processor
        self.benchmark_name = benchmark_name

        logger.info(f"Loading {benchmark_name} dataset...")

        # Use config from dataloader module (single source of truth)
        if benchmark_name not in BENCHMARK_CONFIGS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(BENCHMARK_CONFIGS.keys())}")

        config = BENCHMARK_CONFIGS[benchmark_name]
        self.dataset = load_dataset(config["hf_name"], split=config["split"], trust_remote_code=True)

        # Store field keys from config
        self._question_key = config["question_key"]
        self._answer_key = config["answer_key"]
        self._image_key = config["image_key"]

        # Limit samples if specified
        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))

        logger.info(f"Loaded {len(self.dataset)} samples from {benchmark_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract image using dataloader method (will raise KeyError if not found)
        if self._image_key in item and item[self._image_key] is not None:
            image = item[self._image_key]
        elif 'image' in item and item['image'] is not None:
            image = item['image']
        else:
            raise KeyError(f"No image found in item. Tried keys: {self._image_key}, image. Item keys: {list(item.keys())}")

        # Use fallback chain: let processor handle if ≤1920px, else resize
        image = prepare_image_with_fallback(image, f"benchmark_{self.benchmark_name}_{idx}")

        # Extract question using dataloader method (will raise KeyError if not found)
        question = BenchmarkMixin.extract_question(item, self._question_key)

        # Extract answer using dataloader method (will raise KeyError if not found)
        answer = BenchmarkMixin.extract_answer(item, self._answer_key)

        # Format using chat template - separate prompt and full text for proper masking
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
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

        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        # Process prompt separately to get its length
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Process full text
        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Mask prompt tokens - only train on the answer
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_cache_dir = get_hf_cache_dir()

    def load_model(self, base_model: str = None):
        """Load model with QLoRA for fine-tuning using shared model_utils"""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", BASE_MODEL)

        # Resolve cache directory using shared utility
        cache_dir = resolve_cache_dir(
            self.config.get("model", {}).get("cache_dir", None),
            self.config
        )

        # Use shared model loading function
        self.model, self.processor = load_model_qlora(
            base_model=base_model,
            cache_dir=cache_dir
        )

    def train_qcm(self, dataset_path: str, image_dir: str, output_dir: str,
                  epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                  base_model: str = None, strategy_name: str = "qcm") -> str:
        """Train on QCM dataset"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        # Extract dataset name for logging
        dataset_name = Path(dataset_path).stem  # e.g., "qcm_procedure1_claude_code"

        # Initialize dual logger (WandB offline + TensorBoard)
        # Include training type (qlora) and dataset in tensorboard path
        tensorboard_dir = f"tensorboard_logs/qlora_{strategy_name}_{dataset_name}"
        dual_logger = init_dual_logger(tensorboard_dir, use_wandb=use_wandb and WANDB_AVAILABLE)

        logger.info(f"[QLORA] Training on QCM dataset: {dataset_path}")
        logger.info(f"[QLORA] Strategy: {strategy_name}, Dataset: {dataset_name}")

        # Create dataset
        full_dataset = QCMDataset(dataset_path, image_dir, self.processor)

        # Limit dataset size if max_samples specified
        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        # Split dataset
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("training", {}).get("learning_rate", 1e-5),
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # Extract training dataset name from path (e.g., "qcm_gemini" from "datasets/erp/qcm_gemini.json")
        training_dataset_name = Path(dataset_path).stem

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_dataset_name=training_dataset_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_qcm_combined(self, dataset_paths: list, image_dir: str, output_dir: str,
                           epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                           base_model: str = None, strategy_name: str = "qcm_combined") -> str:
        """Train on combined QCM datasets (Gemini + Nova)"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        # Extract dataset names for logging
        dataset_names = "_".join([Path(p).stem for p in dataset_paths])

        # Initialize dual logger (WandB offline + TensorBoard)
        # Include training type (qlora) and datasets in tensorboard path
        tensorboard_dir = f"tensorboard_logs/qlora_{strategy_name}_combined"
        dual_logger = init_dual_logger(tensorboard_dir, use_wandb=use_wandb and WANDB_AVAILABLE)

        logger.info(f"[QLORA] Training on combined QCM datasets: {dataset_paths}")
        logger.info(f"[QLORA] Strategy: {strategy_name}, Datasets: {dataset_names}")

        # Load all datasets and concatenate
        datasets = []
        for dataset_path in dataset_paths:
            ds = QCMDataset(dataset_path, image_dir, self.processor)
            datasets.append(ds)
            logger.info(f"  Loaded {len(ds)} samples from {Path(dataset_path).name}")

        # Combine datasets
        full_dataset = torch.utils.data.ConcatDataset(datasets)
        logger.info(f"Combined dataset: {len(full_dataset)} total samples")

        # Limit dataset size if max_samples specified
        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        # Split dataset
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("training", {}).get("learning_rate", 1e-5),
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # For combined training, we can't skip individual dataset evaluations
        # because train/test split was done on the combined dataset
        training_dataset_name = None

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_dataset_name=training_dataset_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_chosen_rej_sft(self, dataset_path: str, image_dir: str, output_dir: str,
                      epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                      base_model: str = None, strategy_name: str = "chosen_rej_sft") -> str:
        """Train on chosen/rejected dataset using SFT (chosen responses only)"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        logger.info(f"Training SFT on chosen/rejected dataset: {dataset_path}")

        # Create dataset
        full_dataset = DPOSFTDataset(dataset_path, image_dir, self.processor)

        # Limit dataset size if max_samples specified
        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        # Split dataset
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("training", {}).get("learning_rate", 1e-5),
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # Extract training dataset name from path
        training_dataset_name = Path(dataset_path).stem

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_dataset_name=training_dataset_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_chosen_rej_sft_combined(self, dataset_paths: list, image_dir: str, output_dir: str,
                               epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                               base_model: str = None, strategy_name: str = "chosen_rej_sft_combined") -> str:
        """Train on combined chosen/rejected datasets using SFT (chosen responses only)"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        logger.info(f"Training SFT on combined chosen/rejected datasets: {dataset_paths}")

        # Load all datasets and concatenate
        datasets = []
        for dataset_path in dataset_paths:
            ds = DPOSFTDataset(dataset_path, image_dir, self.processor)
            datasets.append(ds)
            logger.info(f"  Loaded {len(ds)} samples from {Path(dataset_path).name}")

        # Combine datasets
        full_dataset = torch.utils.data.ConcatDataset(datasets)
        logger.info(f"Combined dataset: {len(full_dataset)} total samples")

        # Limit dataset size if max_samples specified
        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        # Split dataset
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("training", {}).get("learning_rate", 1e-5),
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # For combined training, we can't skip individual dataset evaluations
        training_dataset_name = None

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_dataset_name=training_dataset_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_benchmark(self, benchmark_name: str, output_dir: str,
                        epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                        strategy_name: str = None) -> str:
        """Train on a benchmark dataset (DocVQA, OCRBench, ChartQA)"""
        if self.model is None:
            self.load_model()

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name or f"sft_{benchmark_name}",
                config={"base_model": self.config.get("model", {}).get("base_model", "unknown")},
                reinit=True
            )

        logger.info(f"Training on benchmark: {benchmark_name}")

        # Create dataset
        full_dataset = BenchmarkDataset(benchmark_name, self.processor, max_samples)

        # Split dataset
        dataset_size = len(full_dataset)
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("training", {}).get("learning_rate", 1e-5),
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
            report_to="wandb" if use_wandb else "none",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # Use benchmark name to skip duplicate evaluation (e.g., "docvqa", "ocrbench", "chartqa")
        training_dataset_name = benchmark_name

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name or f"sft_{benchmark_name}",
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_dataset_name=training_dataset_name
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Model saved to: {output_dir}")
        return output_dir


def train_sft(config: Dict[str, Any], strategy: Dict[str, Any], output_dir: str,
              base_model: str = None) -> str:
    """
    Train a model using SFT

    Args:
        config: Full configuration
        strategy: Training strategy from config
        output_dir: Where to save the model
        base_model: Optional base model path (for multi-stage training)

    Returns:
        Path to trained model
    """
    trainer = SFTTrainer(config)
    strategy_name = strategy.get("name", strategy["type"])

    if strategy["type"] == "sft_qcm":
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / strategy["dataset"]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_qcm(
            dataset_path=str(dataset_path),
            image_dir=str(image_dir),
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            base_model=base_model,
            strategy_name=strategy_name
        )

    if strategy["type"] == "sft_qcm_combined":
        base_path = Path(__file__).parent.parent
        dataset_paths = [str(base_path / d) for d in strategy["datasets"]]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_qcm_combined(
            dataset_paths=dataset_paths,
            image_dir=str(image_dir),
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            base_model=base_model,
            strategy_name=strategy_name
        )

    if strategy["type"] == "sft_benchmark":
        benchmark_name = strategy.get("benchmark")
        if not benchmark_name:
            raise ValueError("sft_benchmark strategy requires 'benchmark' field")

        return trainer.train_benchmark(
            benchmark_name=benchmark_name,
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            strategy_name=strategy_name
        )

    if strategy["type"] == "sft_chosen_rej":
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / strategy["dataset"]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_chosen_rej_sft(
            dataset_path=str(dataset_path),
            image_dir=str(image_dir),
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            base_model=base_model,
            strategy_name=strategy_name
        )

    if strategy["type"] == "sft_chosen_rej_combined":
        base_path = Path(__file__).parent.parent
        dataset_paths = [str(base_path / d) for d in strategy["datasets"]]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_chosen_rej_sft_combined(
            dataset_paths=dataset_paths,
            image_dir=str(image_dir),
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            base_model=base_model,
            strategy_name=strategy_name
        )

    raise ValueError(f"Unknown training type: {strategy['type']}")
