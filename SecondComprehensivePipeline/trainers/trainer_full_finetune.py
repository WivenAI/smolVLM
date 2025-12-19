"""
Full Fine-tuning Trainer for SmolVLM

Full fine-tuning trains ALL model parameters (no LoRA/QLoRA).
This is recommended for smaller models like SmolVLM-256M and SmolVLM-500M
since they are small enough to fine-tune entirely on modest hardware.

WARNING: Full fine-tuning requires significantly more GPU memory than LoRA.
- SmolVLM-256M: ~4-6 GB VRAM minimum
- SmolVLM-500M: ~8-12 GB VRAM minimum
- SmolVLM-2B: ~24-32 GB VRAM minimum (consider LoRA instead)

Memory optimization tips:
- Use gradient checkpointing (enabled by default)
- Use bf16/fp16 mixed precision
- Use smaller batch size with gradient accumulation
- Use DeepSpeed ZeRO for multi-GPU training
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

# Set HuggingFace cache before imports
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
from datasets import Dataset, load_dataset
import gc

# Import the evaluation callbacks to reuse them
from trainers.trainer_sft import EpochEvaluationCallback
from trainers.trainer_dpo import EpochEvaluationCallback as DPOEpochEvaluationCallback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VisionLanguageDataCollator:
    """Custom data collator for vision-language models"""

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pixel_values = [f.pop('pixel_values') for f in features]
        max_length = max(f['input_ids'].shape[0] for f in features)

        batch = {}
        batch['pixel_values'] = torch.stack(pixel_values)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = f['input_ids'].shape[0]
            pad_len = max_length - seq_len

            input_ids.append(torch.cat([
                f['input_ids'],
                torch.full((pad_len,), 0, dtype=f['input_ids'].dtype)
            ]))

            attention_mask.append(torch.cat([
                f['attention_mask'],
                torch.zeros(pad_len, dtype=f['attention_mask'].dtype)
            ]))

            labels.append(torch.cat([
                f['labels'],
                torch.full((pad_len,), -100, dtype=f['labels'].dtype)
            ]))

        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['labels'] = torch.stack(labels)

        return batch


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
                image = Image.open(image_path).convert('RGB')
            else:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.new('RGB', (224, 224), color='white')

        # Format prompt - item is already the QCM data
        question = item['question']
        options = item['options']
        correct_answer = item['correct_answer']

        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter of the correct option:"

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
                image = Image.open(image_path).convert('RGB')
                max_size = 1024
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            else:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.new('RGB', (224, 224), color='white')

        prompt = item['prompt']
        chosen_response = item['chosen']

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

        # Mask prompt tokens
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


class BenchmarkDataset(torch.utils.data.Dataset):
    """Dataset for training on benchmark datasets"""

    def __init__(self, benchmark_name: str, processor, max_samples: int = None):
        self.processor = processor
        self.benchmark_name = benchmark_name

        logger.info(f"Loading {benchmark_name} dataset...")

        if benchmark_name == "docvqa":
            self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "ocrbench":
            self.dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        elif benchmark_name == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        if max_samples and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))

        logger.info(f"Loaded {len(self.dataset)} samples from {benchmark_name}")

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

        if image.mode != 'RGB':
            image = image.convert('RGB')

        max_size = 1024
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

        # Format using chat template with proper prompt masking
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

        # Mask prompt tokens so we only compute loss on the answer
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)

        return inputs


class FullFineTuneTrainer:
    """
    Full fine-tuning trainer for SmolVLM.

    Unlike LoRA/QLoRA which freeze most parameters and only train adapter weights,
    full fine-tuning trains ALL model parameters. This typically yields better
    results for smaller models but requires more memory.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_cache_dir = _hf_cache

    def load_model(self, base_model: str = None):
        """
        Load model for full fine-tuning (no quantization, no LoRA).

        All model parameters will be trainable.
        """
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", "HuggingFaceTB/SmolVLM-500M-Instruct")

        logger.info(f"Loading model for FULL fine-tuning: {base_model}")

        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        # Load model WITHOUT quantization for full fine-tuning
        # Use bfloat16 for memory efficiency while maintaining training precision
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Full fine-tuning - ALL parameters trainable:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def _get_training_args(self, output_dir: str, epochs: int, use_wandb: bool) -> TrainingArguments:
        """Get training arguments optimized for full fine-tuning."""

        # Full fine-tuning typically needs lower learning rate than LoRA
        # since we're updating all parameters
        full_ft_lr = self.config.get("training", {}).get("full_finetune_learning_rate", 5e-6)

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            # Larger gradient accumulation to compensate for small batch size
            gradient_accumulation_steps=self.config.get("training", {}).get("gradient_accumulation_steps", 16),
            learning_rate=full_ft_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,  # 10% warmup
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
            # Use standard AdamW for full fine-tuning (not 8-bit)
            optim="adamw_torch",
            # Helps with stability during full fine-tuning
            max_grad_norm=1.0,
        )

    def train_qcm(self, dataset_path: str, image_dir: str, output_dir: str,
                  epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                  base_model: str = None, strategy_name: str = "full_ft_qcm") -> str:
        """Train on QCM dataset with full fine-tuning."""
        if self.model is None:
            self.load_model(base_model)

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Full fine-tuning on QCM dataset: {dataset_path}")

        full_dataset = QCMDataset(dataset_path, image_dir, self.processor)

        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        training_args = self._get_training_args(output_dir, epochs, use_wandb)

        # Create evaluation callback to run proper evaluation at each epoch
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
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

        # Save the full model (all weights, not just adapters)
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Full fine-tuned model saved to: {output_dir}")
        return output_dir

    def train_qcm_combined(self, dataset_paths: list, image_dir: str, output_dir: str,
                           epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                           base_model: str = None, strategy_name: str = "full_ft_qcm_combined") -> str:
        """Train on combined QCM datasets with full fine-tuning."""
        if self.model is None:
            self.load_model(base_model)

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Full fine-tuning on combined QCM datasets: {dataset_paths}")

        datasets = []
        for dataset_path in dataset_paths:
            ds = QCMDataset(dataset_path, image_dir, self.processor)
            datasets.append(ds)
            logger.info(f"  Loaded {len(ds)} samples from {Path(dataset_path).name}")

        full_dataset = torch.utils.data.ConcatDataset(datasets)
        logger.info(f"Combined dataset: {len(full_dataset)} total samples")

        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        training_args = self._get_training_args(output_dir, epochs, use_wandb)

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
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

        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Full fine-tuned model saved to: {output_dir}")
        return output_dir

    def train_chosen_rej_sft(self, dataset_path: str, image_dir: str, output_dir: str,
                      epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                      base_model: str = None, strategy_name: str = "full_ft_chosen_rej_sft") -> str:
        """Train on chosen/rejected dataset using SFT with full fine-tuning."""
        if self.model is None:
            self.load_model(base_model)

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Full fine-tuning SFT on chosen/rejected dataset: {dataset_path}")

        full_dataset = DPOSFTDataset(dataset_path, image_dir, self.processor)

        dataset_size = len(full_dataset)
        if max_samples and max_samples < dataset_size:
            logger.info(f"Limiting dataset from {dataset_size} to {max_samples} samples")
            indices = list(range(max_samples))
            full_dataset = torch.utils.data.Subset(full_dataset, indices)
            dataset_size = max_samples

        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        training_args = self._get_training_args(output_dir, epochs, use_wandb)

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
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

        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Full fine-tuned model saved to: {output_dir}")
        return output_dir

    def train_benchmark(self, benchmark_name: str, output_dir: str,
                        epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                        strategy_name: str = None) -> str:
        """Train on a benchmark dataset with full fine-tuning."""
        if self.model is None:
            self.load_model()

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name or f"full_ft_{benchmark_name}",
                reinit=True
            )

        logger.info(f"Full fine-tuning on benchmark: {benchmark_name}")

        full_dataset = BenchmarkDataset(benchmark_name, self.processor, max_samples)

        dataset_size = len(full_dataset)
        train_size = int(0.9 * dataset_size)
        eval_size = dataset_size - train_size

        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        training_args = self._get_training_args(output_dir, epochs, use_wandb)

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name or f"full_ft_{benchmark_name}",
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
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

        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        logger.info(f"Full fine-tuned model saved to: {output_dir}")
        return output_dir


class FullFineTuneDPOTrainer:
    """
    Full fine-tuning DPO trainer for SmolVLM.

    Unlike LoRA DPO which freezes most parameters and only trains adapter weights,
    full fine-tuning DPO trains ALL model parameters during preference optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.ref_model = None
        self.processor = None
        self.hf_cache_dir = _hf_cache

    def load_model(self, base_model: str = None):
        """Load model for full fine-tuning DPO (no quantization, no LoRA)."""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", "HuggingFaceTB/SmolVLM-500M-Instruct")

        logger.info(f"Loading model for FULL fine-tuning DPO: {base_model}")

        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        # Load model WITHOUT quantization for full fine-tuning
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Full fine-tuning DPO - ALL parameters trainable:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    def prepare_dpo_dataset(self, dataset_path: str, image_dir: str, max_samples: int = None) -> Dataset:
        """Prepare DPO dataset from JSON file with actual image loading for VLM DPO"""
        logger.info(f"Preparing DPO dataset from: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        image_dir = Path(image_dir)

        dpo_data = []
        skipped = 0
        for item in raw_data:
            image_name = item.get('image_name', '')
            image = None

            if image_name:
                image_path = image_dir / image_name
                if image_path.exists():
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image.thumbnail((384, 384))
                    except Exception as e:
                        logger.warning(f"Failed to load image {image_path}: {e}")
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue
            else:
                image = Image.new('RGB', (384, 384), color='black')

            prompt = item.get('prompt', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')

            if prompt and chosen and rejected and image:
                prompt_with_image = f"<image>{prompt}"
                dpo_data.append({
                    'prompt': prompt_with_image,
                    'chosen': chosen,
                    'rejected': rejected,
                    'images': [image]
                })

        if max_samples is not None and len(dpo_data) > max_samples:
            logger.info(f"Limiting dataset from {len(dpo_data)} to {max_samples} samples")
            dpo_data = dpo_data[:max_samples]

        if skipped > 0:
            logger.warning(f"Skipped {skipped} samples due to missing/invalid images")

        logger.info(f"Prepared {len(dpo_data)} DPO samples with images")
        return Dataset.from_list(dpo_data)

    def train(self, dataset_path: str, image_dir: str, output_dir: str,
              use_wandb: bool = True, max_samples: int = None,
              strategy_name: str = "full_ft_dpo") -> str:
        """Train using DPO with full fine-tuning."""
        if self.model is None:
            self.load_model()

        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Full fine-tuning DPO on: {dataset_path}")

        full_dataset = self.prepare_dpo_dataset(dataset_path, image_dir, max_samples=max_samples)

        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        # Use lower learning rate for full fine-tuning DPO
        learning_rate = float(self.config.get("training", {}).get("full_finetune_dpo_learning_rate",
                              self.config.get("training", {}).get("dpo_learning_rate", 1e-7)))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 16))

        training_args = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            beta=0.1,
            loss_type="sigmoid",
            max_length=512,
            max_prompt_length=256,
            gradient_checkpointing=True,
            # Use standard optimizer for full fine-tuning
            optim="adamw_torch",
            max_grad_norm=1.0,
        )

        # Create DPO evaluation callback
        eval_callback = DPOEpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,  # Use implicit reference model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.processor,
            callbacks=[eval_callback],
        )

        trainer.train()

        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Full fine-tuned DPO model saved to: {output_dir}")
        return output_dir


def train_full_finetune(config: Dict[str, Any], strategy: Dict[str, Any], output_dir: str,
                        base_model: str = None) -> str:
    """
    Train a model using full fine-tuning (no LoRA).

    Args:
        config: Full configuration
        strategy: Training strategy from config
        output_dir: Where to save the model
        base_model: Optional base model path (for multi-stage training)

    Returns:
        Path to trained model
    """
    trainer = FullFineTuneTrainer(config)
    strategy_name = strategy.get("name", strategy["type"])

    if strategy["type"] == "full_ft_qcm":
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

    if strategy["type"] == "full_ft_qcm_combined":
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

    if strategy["type"] == "full_ft_benchmark":
        benchmark_name = strategy.get("benchmark")
        if not benchmark_name:
            raise ValueError("full_ft_benchmark strategy requires 'benchmark' field")

        return trainer.train_benchmark(
            benchmark_name=benchmark_name,
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            strategy_name=strategy_name
        )

    if strategy["type"] == "full_ft_chosen_rej_sft":
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

    # Full fine-tuning DPO (preference optimization, not SFT)
    if strategy["type"] == "full_ft_dpo":
        dpo_trainer = FullFineTuneDPOTrainer(config)
        dpo_trainer.load_model(base_model)

        base_path = Path(__file__).parent.parent
        dataset_path = base_path / strategy["dataset"]
        image_dir = base_path / strategy["image_dir"]

        return dpo_trainer.train(
            dataset_path=str(dataset_path),
            image_dir=str(image_dir),
            output_dir=output_dir,
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            strategy_name=strategy_name
        )

    raise ValueError(f"Unknown full fine-tuning type: {strategy['type']}")


if __name__ == "__main__":
    # Simple test/demo
    import argparse

    parser = argparse.ArgumentParser(description="Full fine-tune SmolVLM")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--output-dir", type=str, default="./smolvlm-full-finetuned")
    parser.add_argument("--test", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    config = {
        "model": {"base_model": args.base_model},
        "training": {
            "epochs": 1 if args.test else 3,
            "train_samples": 10 if args.test else None,
            "gradient_accumulation_steps": 16,
            "full_finetune_learning_rate": 5e-6,
        },
        "pipeline": {
            "use_wandb": False,
        }
    }

    trainer = FullFineTuneTrainer(config)
    trainer.load_model(args.base_model)

    print("\nModel loaded successfully for full fine-tuning!")
    print("To run training, use the pipeline.py with full_ft_* strategies enabled.")
