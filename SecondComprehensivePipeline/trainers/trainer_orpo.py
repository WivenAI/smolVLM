"""
ORPO Trainer - Odds Ratio Preference Optimization for SmolVLM

ORPO is a reference-free preference optimization method that combines SFT and
preference alignment in a single training step. It's more memory-efficient than
DPO because it doesn't require loading a reference model.

Key advantages over DPO:
- ~50% less GPU memory (no reference model)
- Faster training (single-stage vs multi-stage)
- Same data format as DPO (prompt, chosen, rejected)

NOTE: TRL's ORPOTrainer does NOT officially support VLMs with images (unlike DPOTrainer).
This implementation uses text-only training. For VLM preference optimization, use DPO instead.

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
    BitsAndBytesConfig,
    TrainerCallback
)
from trl import ORPOTrainer as TRLORPOTrainer, ORPOConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochEvaluationCallback(TrainerCallback):
    """Callback to run full evaluation at the end of each epoch"""

    def __init__(self, config: Dict[str, Any], output_dir: str, strategy_name: str, processor):
        self.config = config
        self.output_dir = Path(output_dir)
        self.strategy_name = strategy_name
        self.processor = processor
        self.cache_dir = Path(__file__).parent.parent / "datasets" / "cache"

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run full evaluation at end of each epoch"""
        epoch = int(state.epoch)
        logger.info(f"[{self.strategy_name}] Running evaluation at epoch {epoch}...")

        # Save model temporarily for evaluation
        temp_model_dir = self.output_dir / f"epoch_{epoch}_eval"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        try:
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

                wandb.log(metrics, step=state.global_step)
                logger.info(f"[{self.strategy_name}] Epoch {epoch} eval metrics logged to WandB")

            # Log summary
            logger.info(f"[{self.strategy_name}] Epoch {epoch} evaluation complete:")
            for key, value in results.get("summary", {}).items():
                if "accuracy" in key:
                    logger.info(f"  {key}: {value:.2f}%")

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Evaluation failed at epoch {epoch}: {e}")

        return control


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
        Prepare ORPO dataset from JSON file with actual image loading.

        ORPO uses the same format as DPO:
        - prompt: The input prompt/question
        - chosen: The preferred response
        - rejected: The non-preferred response
        - images: List of PIL Image objects (for VLM support)
        """
        logger.info(f"Preparing ORPO dataset from: {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        image_dir = Path(image_dir)

        # Convert to ORPO format with images column (for VLM support)
        orpo_data = []
        for item in raw_data:
            image_name = item.get('image_name', '')
            image = None

            if image_name:
                image_path = image_dir / image_name
                if image_path.exists():
                    try:
                        image = Image.open(image_path).convert('RGB')
                        # Resize to prevent OOM (SmolVLM uses 384x384)
                        image.thumbnail((384, 384))
                    except Exception as e:
                        logger.warning(f"Failed to load image {image_path}: {e}")
                        continue
                else:
                    continue
            else:
                # Create black placeholder for text-only samples
                image = Image.new('RGB', (384, 384), color='black')

            prompt = item.get('prompt', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')

            if prompt and chosen and rejected and image:
                orpo_data.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'images': [image]  # TRL expects 'images' column with list of PIL Images
                })

        # Apply sample limit if specified
        if max_samples is not None and len(orpo_data) > max_samples:
            logger.info(f"Limiting dataset from {len(orpo_data)} to {max_samples} samples")
            orpo_data = orpo_data[:max_samples]

        logger.info(f"Prepared {len(orpo_data)} ORPO samples with images")
        return Dataset.from_list(orpo_data)

    def train(self, dataset_path: str, image_dir: str, output_dir: str,
              use_wandb: bool = True, max_samples: int = None,
              strategy_name: str = "orpo") -> str:
        """
        Train using ORPO.

        ORPO combines SFT and preference optimization:
        - SFT loss on chosen responses
        - Odds ratio loss to contrast chosen vs rejected

        The beta parameter controls the strength of preference learning.
        """
        if self.model is None:
            self.load_model()

        self.strategy_name = strategy_name

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

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
        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        learning_rate = float(self.config.get("training", {}).get("learning_rate", 5e-6))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 4))

        training_args = ORPOConfig(
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
            eval_strategy="epoch",
            save_strategy="epoch",
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

        # Create evaluation callback
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor
        )

        trainer = TRLORPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[eval_callback],
        )

        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        # Finish WandB run
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

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
    strategy_name = strategy.get("name", "orpo")

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
