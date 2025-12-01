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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

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

        # Format prompt
        qcm_data = item.get('qcm', item)
        question = qcm_data['question']
        options = qcm_data['options']
        correct_answer = qcm_data['correct_answer']
        explanation = qcm_data.get('explanation', '')

        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"{question}\n\nOptions:\n{options_text}\n\nAnswer:"

        correct_option_text = options[correct_answer]
        answer = f"{correct_answer} - {correct_option_text}"
        if explanation:
            answer += f"\n\nExplanation: {explanation}"

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


class BenchmarkDataset(torch.utils.data.Dataset):
    """Dataset for training on benchmark datasets (DocVQA, OCRBench, ChartQA)"""

    def __init__(self, benchmark_name: str, processor, max_samples: int = None):
        self.processor = processor
        self.benchmark_name = benchmark_name

        logger.info(f"Loading {benchmark_name} dataset...")

        # Load different benchmarks
        if benchmark_name == "docvqa":
            self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "ocrbench":
            self.dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        elif benchmark_name == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Limit samples if specified
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

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize large images
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

        # Format using chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        full_text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

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

        # Get image token ID
        image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]

        # Clone input_ids for labels
        inputs["labels"] = inputs["input_ids"].clone()

        # Mask padding tokens
        inputs["labels"][inputs["labels"] == self.processor.tokenizer.pad_token_id] = -100

        # Mask image tokens
        inputs["labels"][inputs["labels"] == image_token_id] = -100

        return inputs


class SFTTrainer:
    """Trainer for Supervised Fine-Tuning"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_cache_dir = _hf_cache

    def load_model(self, base_model: str = None):
        """Load model with LoRA for fine-tuning"""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", "HuggingFaceTB/SmolVLM-500M-Instruct")

        logger.info(f"Loading model: {base_model}")

        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train_qcm(self, dataset_path: str, image_dir: str, output_dir: str,
                  epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                  base_model: str = None) -> str:
        """Train on QCM dataset"""
        if self.model is None:
            self.load_model(base_model)

        logger.info(f"Training on QCM dataset: {dataset_path}")

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
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

    def train_benchmark(self, benchmark_name: str, output_dir: str,
                        epochs: int = 3, use_wandb: bool = True, max_samples: int = None) -> str:
        """Train on a benchmark dataset (DocVQA, OCRBench, ChartQA)"""
        if self.model is None:
            self.load_model()

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
            gradient_accumulation_steps=8,
            learning_rate=1e-4,  # Higher LR for benchmark training
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=VisionLanguageDataCollator(),
        )

        trainer.train()

        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)

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
            base_model=base_model
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
            max_samples=config.get("training", {}).get("train_samples")
        )

    raise ValueError(f"Unknown training type: {strategy['type']}")
