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
from datasets import Dataset, load_dataset
import random

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpochEvaluationCallback(TrainerCallback):
    """Callback to run full evaluation at the end of each epoch.

    Evaluates on both train and test sets separately to detect memorization vs overfitting:
    - High train accuracy + low test accuracy = overfitting
    - High train accuracy + high test accuracy = good generalization
    - Low train accuracy = underfitting
    """

    def __init__(self, config: Dict[str, Any], output_dir: str, strategy_name: str, processor,
                 train_dataset=None, eval_dataset=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.strategy_name = strategy_name
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cache_dir = Path(__file__).parent.parent / "datasets" / "cache"

    def _compute_orpo_metrics(self, model, dataset, dataset_name="dataset"):
        """Compute ORPO-specific metrics: preference accuracy (chosen > rejected)"""
        if dataset is None or self.processor is None:
            return None, None

        model.eval()
        correct_preferences = 0
        total = 0
        total_margin = 0.0

        # Get tokenizer from processor
        tokenizer = self.processor if hasattr(self.processor, 'pad_token_id') else self.processor.tokenizer

        with torch.no_grad():
            for idx in range(len(dataset)):
                try:
                    item = dataset[idx]

                    # Get prompt, chosen, rejected
                    prompt = item.get('prompt', '')
                    chosen = item.get('chosen', '')
                    rejected = item.get('rejected', '')

                    if not prompt or not chosen or not rejected:
                        continue

                    device = next(model.parameters()).device

                    # Compute log probs for chosen
                    chosen_text = f"{prompt}{chosen}"
                    chosen_inputs = tokenizer(
                        chosen_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    chosen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in chosen_inputs.items()}
                    chosen_inputs['labels'] = chosen_inputs['input_ids'].clone()
                    chosen_outputs = model(**chosen_inputs)
                    chosen_loss = chosen_outputs.loss.item()

                    # Compute log probs for rejected
                    rejected_text = f"{prompt}{rejected}"
                    rejected_inputs = tokenizer(
                        rejected_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    rejected_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                      for k, v in rejected_inputs.items()}
                    rejected_inputs['labels'] = rejected_inputs['input_ids'].clone()
                    rejected_outputs = model(**rejected_inputs)
                    rejected_loss = rejected_outputs.loss.item()

                    # Lower loss = higher probability = preferred
                    if chosen_loss < rejected_loss:
                        correct_preferences += 1

                    margin = rejected_loss - chosen_loss
                    total_margin += margin
                    total += 1

                except Exception as e:
                    logger.warning(f"Error computing ORPO metrics for sample {idx} in {dataset_name}: {e}")
                    continue

        model.train()

        if total > 0:
            accuracy = (correct_preferences / total) * 100
            avg_margin = total_margin / total
            return accuracy, avg_margin
        return None, None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run full evaluation at end of each epoch on both train and test sets"""
        epoch = int(state.epoch)
        logger.info(f"[{self.strategy_name}] Running train/test evaluation at epoch {epoch}...")

        # Save model temporarily for evaluation
        temp_model_dir = self.output_dir / f"epoch_{epoch}_eval"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute ORPO preference accuracy on train and test sets
            train_pref_acc, train_margin = self._compute_orpo_metrics(model, self.train_dataset, "train")
            test_pref_acc, test_margin = self._compute_orpo_metrics(model, self.eval_dataset, "test")

            # Log results
            logger.info(f"[{self.strategy_name}] Epoch {epoch} Train/Test ORPO Metrics:")
            if train_pref_acc is not None:
                logger.info(f"  Train Preference Accuracy: {train_pref_acc:.2f}% (margin: {train_margin:.4f})")
            if test_pref_acc is not None:
                logger.info(f"  Test Preference Accuracy: {test_pref_acc:.2f}% (margin: {test_margin:.4f})")

            # Check for memorization/overfitting
            if train_pref_acc is not None and test_pref_acc is not None:
                gap = train_pref_acc - test_pref_acc
                if gap > 10:
                    logger.warning(f"  ⚠️ Large train-test gap ({gap:.2f}%): possible OVERFITTING")
                elif train_pref_acc > 90 and test_pref_acc > 80:
                    logger.info(f"  ✓ Good generalization (train: {train_pref_acc:.1f}%, test: {test_pref_acc:.1f}%)")
                elif train_pref_acc < 60:
                    logger.warning(f"  ⚠️ Low train preference accuracy ({train_pref_acc:.1f}%): ORPO not learning preferences")

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

                # Log train/test ORPO metrics
                if train_pref_acc is not None:
                    metrics["eval/train_preference_acc"] = train_pref_acc
                    metrics["eval/train_margin"] = train_margin
                if test_pref_acc is not None:
                    metrics["eval/test_preference_acc"] = test_pref_acc
                    metrics["eval/test_margin"] = test_margin

                # Log train-test gap (for memorization detection)
                if train_pref_acc is not None and test_pref_acc is not None:
                    metrics["eval/train_test_gap"] = train_pref_acc - test_pref_acc

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
            import traceback
            traceback.print_exc()

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
        skipped_missing_image = 0
        skipped_no_image_name = 0
        skipped_load_error = 0
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
                        skipped_load_error += 1
                        continue
                else:
                    logger.debug(f"Image not found: {image_path}")
                    skipped_missing_image += 1
                    continue
            else:
                logger.debug(f"No image_name in item, using placeholder")
                skipped_no_image_name += 1
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

        # Log summary of skipped samples
        total_skipped = skipped_missing_image + skipped_no_image_name + skipped_load_error
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} samples: {skipped_missing_image} missing images, "
                          f"{skipped_load_error} load errors, {skipped_no_image_name} no image_name")

        logger.info(f"Prepared {len(orpo_data)} ORPO samples with images")
        return Dataset.from_list(orpo_data)

    def prepare_benchmark_orpo_dataset(self, benchmark_name: str, max_samples: int = None) -> Dataset:
        """Prepare ORPO dataset from benchmark by using correct answer as chosen and random wrong answer as rejected"""
        logger.info(f"Preparing ORPO dataset from benchmark: {benchmark_name}")

        # Load benchmark dataset
        if benchmark_name == "docvqa":
            dataset = load_dataset("nielsr/docvqa_1200_examples", split="train", trust_remote_code=True)
        elif benchmark_name == "ocrbench":
            dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        elif benchmark_name == "chartqa":
            dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # Limit samples if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        # Collect all answers for generating wrong answers
        all_answers = []
        for item in dataset:
            if 'answers' in item:
                answers = item['answers']
                if isinstance(answers, list) and len(answers) > 0:
                    all_answers.append(answers[0])
                else:
                    all_answers.append(str(answers))
            elif 'answer' in item:
                all_answers.append(item['answer'])
            elif 'label' in item:
                all_answers.append(str(item['label']))

        # Convert to ORPO format with images (experimental VLM support)
        orpo_data = []
        skipped_no_image = 0
        skipped_no_answer = 0
        for idx, item in enumerate(dataset):
            # Extract image
            if 'image' in item:
                image = item['image']
            elif 'img' in item:
                image = item['img']
            else:
                logger.debug(f"Sample {idx}: No image field found")
                skipped_no_image += 1
                continue

            # Convert to RGB and resize
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.thumbnail((384, 384))

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

            # Extract correct answer (chosen)
            if 'answers' in item:
                answers = item['answers']
                if isinstance(answers, list) and len(answers) > 0:
                    chosen = answers[0]
                else:
                    chosen = str(answers)
            elif 'answer' in item:
                chosen = item['answer']
            elif 'label' in item:
                chosen = str(item['label'])
            else:
                logger.debug(f"Sample {idx}: No answer field found")
                skipped_no_answer += 1
                continue

            # Generate rejected answer (random wrong answer from other samples)
            rejected_candidates = [a for a in all_answers if a != chosen]
            if not rejected_candidates:
                rejected = "I don't know"
            else:
                rejected = random.choice(rejected_candidates)

            # Format prompt with image token for VLM
            prompt = f"<image>Answer briefly. {question}"

            orpo_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'images': [image]  # Add images column for VLM support
            })

        # Log summary of skipped samples
        total_skipped = skipped_no_image + skipped_no_answer
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} samples from {benchmark_name}: "
                          f"{skipped_no_image} no image, {skipped_no_answer} no answer")

        logger.info(f"Prepared {len(orpo_data)} ORPO samples with images from {benchmark_name}")
        return Dataset.from_list(orpo_data)

    def train_benchmark(self, benchmark_name: str, output_dir: str,
                        use_wandb: bool = True, max_samples: int = None,
                        strategy_name: str = "orpo_benchmark") -> str:
        """Train using ORPO on a benchmark dataset (text-only)"""
        if self.model is None:
            self.load_model()

        # Initialize WandB run for this strategy
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.get("pipeline", {}).get("wandb_project", "SmallVLM-NoHallucinations"),
                name=strategy_name,
                reinit=True
            )

        logger.info(f"Training with ORPO on benchmark: {benchmark_name}")

        # Prepare dataset
        full_dataset = self.prepare_benchmark_orpo_dataset(benchmark_name, max_samples=max_samples)

        # Split dataset
        dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset_split['train']
        eval_dataset = dataset_split['test']

        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

        # Get training config values
        num_epochs = int(self.config.get("training", {}).get("epochs", 3))
        learning_rate = float(self.config.get("training", {}).get("learning_rate", 5e-6))
        gradient_accumulation_steps = int(self.config.get("training", {}).get("gradient_accumulation_steps", 4))

        # ORPO config
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
            beta=0.1,
            max_length=512,
            max_prompt_length=256,
            dataset_num_proc=2,
            gradient_checkpointing=True,
            optim="adamw_8bit",
        )

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Use tokenizer for ORPO (it requires pad_token_id which processor doesn't have)
        # Images are still in the dataset - experimental VLM support
        tokenizer = self.processor.tokenizer

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

        logger.info(f"Model saved to: {output_dir}")
        return output_dir

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

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name,
            processor=self.processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
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
