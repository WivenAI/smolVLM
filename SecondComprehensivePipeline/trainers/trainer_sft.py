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
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochEvaluationCallback(TrainerCallback):
    """Callback to run full evaluation at the end of each epoch"""

    def __init__(self, config: Dict[str, Any], output_dir: str, strategy_name: str, processor=None, full_dataset=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.strategy_name = strategy_name
        self.processor = processor
        self.full_dataset = full_dataset
        self.cache_dir = Path(__file__).parent.parent / "datasets" / "cache"

    def _compute_full_dataset_loss(self, model, state):
        """Compute loss on the full dataset"""
        if self.full_dataset is None:
            return None

        model.eval()
        total_loss = 0.0
        num_batches = 0

        # Create a simple dataloader for the full dataset
        from torch.utils.data import DataLoader
        dataloader = DataLoader(self.full_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                try:
                    # Move batch to model device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    outputs = model(**inputs)
                    if hasattr(outputs, 'loss') and outputs.loss is not None:
                        total_loss += outputs.loss.item()
                        num_batches += 1
                except Exception as e:
                    logger.warning(f"Error computing loss for batch: {e}")
                    continue

        model.train()

        if num_batches > 0:
            return total_loss / num_batches
        return None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run full evaluation at end of each epoch"""
        epoch = int(state.epoch)
        logger.info(f"[{self.strategy_name}] Running evaluation at epoch {epoch}...")

        # Save model temporarily for evaluation
        temp_model_dir = self.output_dir / f"epoch_{epoch}_eval"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute loss on full dataset first (before saving model)
            full_dataset_loss = self._compute_full_dataset_loss(model, state)
            if full_dataset_loss is not None:
                logger.info(f"[{self.strategy_name}] Epoch {epoch} full dataset loss: {full_dataset_loss:.4f}")

            # Save the current model state
            model.save_pretrained(str(temp_model_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(temp_model_dir))

            # Import evaluator here to avoid circular imports
            from evaluators import EvaluatorAll

            # Run evaluation
            evaluator = EvaluatorAll(self.config, str(self.cache_dir))
            results = evaluator.evaluate_all(
                model_path=str(temp_model_dir),
                model_name=f"{self.strategy_name}_epoch{epoch}"
            )

            # Log to WandB
            if WANDB_AVAILABLE and wandb.run is not None:
                metrics = {}

                # Log full dataset loss
                if full_dataset_loss is not None:
                    metrics["eval/full_dataset_loss"] = full_dataset_loss

                # Log benchmark accuracies
                for bench_name, bench_data in results.get("benchmarks", {}).items():
                    if "accuracy" in bench_data:
                        metrics[f"eval/{bench_name}_acc"] = bench_data["accuracy"]

                # Log ERP evaluation metrics
                erp = results.get("erp_evaluation", {})
                if "qcm_gemini" in erp and "accuracy" in erp["qcm_gemini"]:
                    metrics["eval/qcm_gemini_acc"] = erp["qcm_gemini"]["accuracy"]
                if "qcm_nova" in erp and "accuracy" in erp["qcm_nova"]:
                    metrics["eval/qcm_nova_acc"] = erp["qcm_nova"]["accuracy"]
                if "qcm_claudette" in erp and "accuracy" in erp["qcm_claudette"]:
                    metrics["eval/qcm_claudette_acc"] = erp["qcm_claudette"]["accuracy"]
                if "dpo_logprobs" in erp and "accuracy" in erp["dpo_logprobs"]:
                    metrics["eval/dpo_logprob_acc"] = erp["dpo_logprobs"]["accuracy"]

                # Log average
                if results.get("summary", {}).get("avg_benchmark_accuracy"):
                    metrics["eval/avg_benchmark_acc"] = results["summary"]["avg_benchmark_accuracy"]

                # Log all metrics at current step
                wandb.log(metrics, step=state.global_step)
                logger.info(f"[{self.strategy_name}] Epoch {epoch} eval metrics logged to WandB")

            # Log summary
            logger.info(f"[{self.strategy_name}] Epoch {epoch} evaluation complete:")
            for key, value in results.get("summary", {}).items():
                if "accuracy" in key:
                    logger.info(f"  {key}: {value:.2f}%")

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Evaluation failed at epoch {epoch}: {e}")

        finally:
            # Clean up temp model (optional - keep for debugging)
            # import shutil
            # shutil.rmtree(temp_model_dir, ignore_errors=True)
            pass

        return control


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
                image = Image.open(image_path).convert('RGB')
                # Resize large images
                max_size = 1024
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            else:
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.new('RGB', (224, 224), color='white')

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
                  base_model: str = None, strategy_name: str = "qcm") -> str:
        """Train on QCM dataset"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            full_dataset=full_dataset
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
                reinit=True
            )

        logger.info(f"Training on combined QCM datasets: {dataset_paths}")

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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            full_dataset=full_dataset
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

    def train_dpo_sft(self, dataset_path: str, image_dir: str, output_dir: str,
                      epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                      base_model: str = None, strategy_name: str = "dpo_sft") -> str:
        """Train on DPO dataset using SFT (chosen responses only)"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Training SFT on DPO dataset: {dataset_path}")

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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            full_dataset=full_dataset
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

    def train_dpo_sft_combined(self, dataset_paths: list, image_dir: str, output_dir: str,
                               epochs: int = 3, use_wandb: bool = True, max_samples: int = None,
                               base_model: str = None, strategy_name: str = "dpo_sft_combined") -> str:
        """Train on combined DPO datasets using SFT (chosen responses only)"""
        if self.model is None:
            self.load_model(base_model)

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Training SFT on combined DPO datasets: {dataset_paths}")

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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            full_dataset=full_dataset
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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name or f"sft_{benchmark_name}",
            processor=self.processor,
            full_dataset=full_dataset
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

    if strategy["type"] == "sft_dpo":
        base_path = Path(__file__).parent.parent
        dataset_path = base_path / strategy["dataset"]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_dpo_sft(
            dataset_path=str(dataset_path),
            image_dir=str(image_dir),
            output_dir=output_dir,
            epochs=config.get("training", {}).get("epochs", 3),
            use_wandb=config.get("pipeline", {}).get("use_wandb", True),
            max_samples=config.get("training", {}).get("train_samples"),
            base_model=base_model,
            strategy_name=strategy_name
        )

    if strategy["type"] == "sft_dpo_combined":
        base_path = Path(__file__).parent.parent
        dataset_paths = [str(base_path / d) for d in strategy["datasets"]]
        image_dir = base_path / strategy["image_dir"]

        return trainer.train_dpo_sft_combined(
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
