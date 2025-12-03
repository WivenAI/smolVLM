"""
ORPO Trainer - Odds Ratio Preference Optimization for SmolVLM

ORPO is a reference-free preference optimization method that combines SFT and
preference alignment in a single training step. It's more memory-efficient than
DPO because it doesn't require loading a reference model.

Key advantages over DPO:
- ~50% less GPU memory (no reference model)
- Faster training (single-stage vs multi-stage)
- Same data format as DPO (prompt, chosen, rejected)

Paper: https://arxiv.org/abs/2403.07691
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import gc
import torch
from PIL import Image

# Set HuggingFace cache before imports
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(_hf_cache, "datasets")

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from trl import ORPOTrainer as TRLORPOTrainer, ORPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ORPOTrainerWrapper:
    """
    Wrapper for ORPO (Odds Ratio Preference Optimization) training.

    ORPO is a reference-free alternative to DPO that:
    - Combines SFT and preference alignment in one step
    - Uses odds ratio to contrast chosen vs rejected responses
    - Requires ~50% less GPU memory than DPO (no reference model)

    Works with 8GB GPUs when combined with QLoRA (4-bit quantization).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_cache_dir = _hf_cache

    def load_model(self, base_model: str = None):
        """Load model with QLoRA for memory-efficient ORPO training"""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", "HuggingFaceTB/SmolVLM-500M-Instruct")

        logger.info(f"Loading model for ORPO: {base_model}")

        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

        # 4-bit quantization for 8GB GPU compatibility
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

        # LoRA config - same as DPO for consistency
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

    def prepare_orpo_dataset(self, dataset_path: str, image_dir: str, max_samples: int = None) -> Dataset:
        """
        Prepare ORPO dataset from JSON file.

        ORPO uses the same format as DPO:
        - prompt: The input prompt/question
        - chosen: The preferred response
        - rejected: The non-preferred response
        """
        logger.info(f"Preparing ORPO dataset from: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        image_dir = Path(image_dir)

        # Convert to ORPO format (same as DPO)
        orpo_data = []
        for item in raw_data:
            image_name = item.get('image_name', '')
            if image_name:
                image_path = image_dir / image_name
                if not image_path.exists():
                    continue
            else:
                image_path = None

            prompt = item.get('prompt', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')

            if prompt and chosen and rejected:
                orpo_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'image_path': str(image_path) if image_path else None
                })

        # Apply sample limit if specified
        if max_samples is not None and len(orpo_data) > max_samples:
            logger.info(f"Limiting dataset from {len(orpo_data)} to {max_samples} samples")
            orpo_data = orpo_data[:max_samples]

        logger.info(f"Prepared {len(orpo_data)} ORPO samples")
        return Dataset.from_list(orpo_data)

    def train(self, dataset_path: str, image_dir: str, output_dir: str,
              use_wandb: bool = True, max_samples: int = None) -> str:
        """
        Train using ORPO.

        ORPO combines SFT and preference optimization:
        - SFT loss on chosen responses
        - Odds ratio loss to contrast chosen vs rejected

        The beta parameter controls the strength of preference learning.
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Training with ORPO on: {dataset_path}")

        # Prepare dataset
        full_dataset = self.prepare_orpo_dataset(dataset_path, image_dir, max_samples=max_samples)

        # Split dataset
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # ORPO config - optimized for 8GB GPU
        # Key differences from DPO:
        # - No reference model needed
        # - beta controls odds ratio strength (typically 0.1)
        training_args = ORPOConfig(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,  # ORPO typically uses lower LR than DPO
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to="wandb" if use_wandb else "none",
            # ORPO-specific parameters
            beta=0.1,  # Odds ratio strength (controls preference learning)
            max_length=512,
            max_prompt_length=256,
            dataset_num_proc=2,
            # Memory optimizations
            gradient_checkpointing=True,
            optim="adamw_8bit",  # 8-bit optimizer for memory savings
        )

        # Use tokenizer instead of processor - ORPOTrainer expects pad_token_id
        # which exists on the tokenizer, not the Idefics3Processor
        tokenizer = self.processor.tokenizer

        trainer = TRLORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"ORPO model saved to: {output_dir}")
        return output_dir


def train_orpo(config: Dict[str, Any], strategy: Dict[str, Any], output_dir: str,
               base_model: str = None) -> str:
    """
    Train a model using ORPO (Odds Ratio Preference Optimization).

    ORPO advantages over DPO:
    - ~50% less GPU memory (no reference model)
    - Single-stage training (combines SFT + preference)
    - Works well with limited data

    Args:
        config: Full configuration
        strategy: Training strategy from config
        output_dir: Where to save the model
        base_model: Base model to start from (can be path to previously trained model)

    Returns:
        Path to trained model
    """
    trainer = ORPOTrainerWrapper(config)

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
        max_samples=config.get("training", {}).get("train_samples")
    )
