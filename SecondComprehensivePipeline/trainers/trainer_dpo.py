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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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


class RAMMonitorCallback(TrainerCallback):
    """Callback to log RAM usage to WandB during training"""

    def __init__(self, log_every_n_steps: int = 10):
        self.log_every_n_steps = log_every_n_steps
        self._initial_ram = get_ram_usage_gb()
        logger.info(f"Initial process RAM: {self._initial_ram:.2f} GB")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            current_ram = get_ram_usage_gb()
            sys_ram = get_system_ram_info()

            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    'system/process_ram_gb': current_ram,
                    'system/ram_delta_gb': current_ram - self._initial_ram,
                    'system/system_ram_used_gb': sys_ram['used_gb'],
                    'system/system_ram_available_gb': sys_ram['available_gb'],
                    'system/system_ram_percent': sys_ram['percent'],
                }, step=state.global_step)

        return control

# Import shared image utilities
from trainers.image_utils import prepare_image_with_fallback


class EpochEvaluationCallback(TrainerCallback):
    """Callback to run full evaluation at the end of each epoch and at specific steps.

    Evaluates on both train and test sets separately to detect memorization vs overfitting:
    - High train accuracy + low test accuracy = overfitting
    - High train accuracy + high test accuracy = good generalization
    - Low train accuracy = underfitting

    Also runs evaluation at steps 0, 5, 10, 50 to detect early training issues.
    """

    # Steps at which to run early evaluation (to detect model breaking)
    EARLY_EVAL_STEPS = [0, 1, 5, 10, 20, 50, 100]

    # Epochs at which to run evaluation (skip 11-19, 21-29)
    EVAL_EPOCHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]

    def __init__(self, config: Dict[str, Any], output_dir: str, strategy_name: str, processor,
                 train_dataset=None, eval_dataset=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.strategy_name = strategy_name
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cache_dir = Path(__file__).parent.parent / "datasets" / "cache"
        self._initial_eval_done = False
        self._evaluated_steps = set()  # Track which steps we've evaluated

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Run baseline evaluation at step 0 before any training."""
        if self._initial_eval_done:
            return control

        self._initial_eval_done = True
        self._evaluated_steps.add(0)
        logger.info(f"[{self.strategy_name}] Running BASELINE evaluation at step 0 (before training)...")

        # Reuse _run_evaluation logic with epoch=0
        self._run_evaluation(args, state, control, model, epoch=0, is_step_eval=True)

        return control

    def _compute_dpo_metrics(self, model, dataset, dataset_name="dataset"):
        """Compute DPO-specific metrics: preference accuracy (chosen > rejected)"""
        if dataset is None or self.processor is None:
            return None, None

        model.eval()
        correct_preferences = 0
        total = 0
        total_margin = 0.0

        with torch.no_grad():
            for idx in range(len(dataset)):
                try:
                    item = dataset[idx]

                    # Get prompt, chosen, rejected
                    prompt = item.get('prompt', '')
                    chosen = item.get('chosen', '')
                    rejected = item.get('rejected', '')
                    images = item.get('images', None)

                    if not prompt or not chosen or not rejected:
                        continue

                    device = next(model.parameters()).device

                    # Compute log probs for chosen
                    chosen_text = f"{prompt}{chosen}"
                    chosen_inputs = self.processor(
                        text=chosen_text,
                        images=images[0] if images else None,
                        return_tensors="pt",
                        padding=True
                    )
                    chosen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in chosen_inputs.items()}
                    chosen_inputs['labels'] = chosen_inputs['input_ids'].clone()
                    chosen_outputs = model(**chosen_inputs)
                    chosen_loss = chosen_outputs.loss.item()

                    # Compute log probs for rejected
                    rejected_text = f"{prompt}{rejected}"
                    rejected_inputs = self.processor(
                        text=rejected_text,
                        images=images[0] if images else None,
                        return_tensors="pt",
                        padding=True
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
                    logger.warning(f"Error computing DPO metrics for sample {idx} in {dataset_name}: {e}")
                    continue

        model.train()

        if total > 0:
            accuracy = (correct_preferences / total) * 100
            avg_margin = total_margin / total
            return accuracy, avg_margin
        return None, None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specific early steps to detect model breaking."""
        current_step = state.global_step

        # Check if this step should trigger evaluation
        if current_step in self.EARLY_EVAL_STEPS and current_step not in self._evaluated_steps:
            self._evaluated_steps.add(current_step)
            logger.info(f"[{self.strategy_name}] Running EARLY evaluation at step {current_step}...")
            self._run_evaluation(args, state, control, model, epoch=current_step, is_step_eval=True)

        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run full evaluation at end of specific epochs on both train and test sets"""
        epoch = int(state.epoch)
        if epoch not in self.EVAL_EPOCHS:
            logger.info(f"[{self.strategy_name}] Skipping evaluation at epoch {epoch} (not in EVAL_EPOCHS)")
            return control
        logger.info(f"[{self.strategy_name}] Running train/test evaluation at epoch {epoch}...")
        self._run_evaluation(args, state, control, model, epoch=epoch, is_step_eval=False)
        return control

    def _run_evaluation(self, args, state, control, model, epoch: int, is_step_eval: bool = False):
        """
        Shared evaluation logic for both step-based and epoch-based evaluation.

        Args:
            epoch: For step eval, this is the step number. For epoch eval, this is the epoch number.
            is_step_eval: If True, only evaluate test set (faster). If False, evaluate both train and test.
        """
        eval_type = "step" if is_step_eval else "epoch"
        temp_model_dir = self.output_dir / f"{eval_type}_{epoch}_eval"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # For step eval, only compute test metrics (faster)
            # For epoch eval, compute both train and test metrics
            train_pref_acc = None
            train_margin = None

            if not is_step_eval:
                train_pref_acc, train_margin = self._compute_dpo_metrics(model, self.train_dataset, "train")

            test_pref_acc, test_margin = self._compute_dpo_metrics(model, self.eval_dataset, "test")

            # Log results
            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} DPO Metrics:")
            if train_pref_acc is not None:
                logger.info(f"  Train Preference Accuracy: {train_pref_acc:.2f}% (margin: {train_margin:.4f})")
            if test_pref_acc is not None:
                logger.info(f"  Test Preference Accuracy: {test_pref_acc:.2f}% (margin: {test_margin:.4f})")

            # Check for memorization/overfitting (only for epoch eval)
            if not is_step_eval and train_pref_acc is not None and test_pref_acc is not None:
                gap = train_pref_acc - test_pref_acc
                if gap > 10:
                    logger.warning(f"  ⚠️ Large train-test gap ({gap:.2f}%): possible OVERFITTING")
                elif train_pref_acc > 90 and test_pref_acc > 80:
                    logger.info(f"  ✓ Good generalization (train: {train_pref_acc:.1f}%, test: {test_pref_acc:.1f}%)")
                elif train_pref_acc < 60:
                    logger.warning(f"  ⚠️ Low train preference accuracy ({train_pref_acc:.1f}%): DPO not learning preferences")

            model.save_pretrained(str(temp_model_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(temp_model_dir))

            from evaluators import EvaluatorAll
            evaluator = EvaluatorAll(self.config, str(self.cache_dir))
            results = evaluator.evaluate_all(
                model_path=str(temp_model_dir),
                model_name=f"{self.strategy_name}_{eval_type}{epoch}"
            )

            if WANDB_AVAILABLE and wandb.run is not None:
                metrics = {}

                # Log train/test DPO metrics
                if train_pref_acc is not None:
                    metrics["eval/train_preference_acc"] = train_pref_acc
                    metrics["eval/train_margin"] = train_margin
                if test_pref_acc is not None:
                    metrics["eval/test_preference_acc"] = test_pref_acc
                    metrics["eval/test_margin"] = test_margin

                # Log train-test gap (for memorization detection)
                if train_pref_acc is not None and test_pref_acc is not None:
                    metrics["eval/train_test_gap"] = train_pref_acc - test_pref_acc

                for bench_name, bench_data in results.get("benchmarks", {}).items():
                    if "accuracy" in bench_data:
                        metrics[f"eval/{bench_name}_acc"] = bench_data["accuracy"]

                erp = results.get("erp_evaluation", {})
                if "qcm_gemini" in erp and "accuracy" in erp["qcm_gemini"]:
                    metrics["eval/qcm_gemini_acc"] = erp["qcm_gemini"]["accuracy"]
                if "qcm_nova" in erp and "accuracy" in erp["qcm_nova"]:
                    metrics["eval/qcm_nova_acc"] = erp["qcm_nova"]["accuracy"]
                if "qcm_claudette" in erp and "accuracy" in erp["qcm_claudette"]:
                    metrics["eval/qcm_claudette_acc"] = erp["qcm_claudette"]["accuracy"]
                if "qcm_procedure1" in erp and "accuracy" in erp["qcm_procedure1"]:
                    metrics["eval/qcm_procedure1_acc"] = erp["qcm_procedure1"]["accuracy"]
                if "qcm_procedure2" in erp and "accuracy" in erp["qcm_procedure2"]:
                    metrics["eval/qcm_procedure2_acc"] = erp["qcm_procedure2"]["accuracy"]

                # Log DPO logprob metrics
                if "dpo_logprobs" in erp and "accuracy" in erp["dpo_logprobs"]:
                    metrics["eval/dpo_logprob_acc"] = erp["dpo_logprobs"]["accuracy"]
                    if "margin_mean" in erp["dpo_logprobs"]:
                        metrics["eval/dpo_logprob_margin"] = erp["dpo_logprobs"]["margin_mean"]

                # Log ROUGE metrics for gemini and nova DPO
                if "rouge_gemini" in erp and "accuracy" in erp["rouge_gemini"]:
                    metrics["eval/rouge_gemini_acc"] = erp["rouge_gemini"]["accuracy"]
                    metrics["eval/rouge_gemini_rouge1"] = erp["rouge_gemini"].get("rouge1", 0)
                    metrics["eval/rouge_gemini_rouge2"] = erp["rouge_gemini"].get("rouge2", 0)
                    metrics["eval/rouge_gemini_rougeL"] = erp["rouge_gemini"].get("rougeL", 0)
                if "rouge_nova" in erp and "accuracy" in erp["rouge_nova"]:
                    metrics["eval/rouge_nova_acc"] = erp["rouge_nova"]["accuracy"]
                    metrics["eval/rouge_nova_rouge1"] = erp["rouge_nova"].get("rouge1", 0)
                    metrics["eval/rouge_nova_rouge2"] = erp["rouge_nova"].get("rouge2", 0)
                    metrics["eval/rouge_nova_rougeL"] = erp["rouge_nova"].get("rougeL", 0)

                # Log BERTScore metrics
                if "bertscore" in erp and "f1" in erp["bertscore"]:
                    metrics["eval/bertscore_f1"] = erp["bertscore"]["f1"]
                    metrics["eval/bertscore_precision"] = erp["bertscore"].get("precision", 0)
                    metrics["eval/bertscore_recall"] = erp["bertscore"].get("recall", 0)

                if results.get("summary", {}).get("avg_benchmark_accuracy"):
                    metrics["eval/avg_benchmark_acc"] = results["summary"]["avg_benchmark_accuracy"]

                # Log epoch/step number explicitly
                metrics[eval_type] = epoch
                metrics["global_step"] = state.global_step

                wandb.log(metrics, step=state.global_step)
                logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} eval metrics logged to WandB")

            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} evaluation complete")

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Evaluation failed at {eval_type} {epoch}: {e}")
            import traceback
            traceback.print_exc()


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
        """Load model with LoRA for DPO training"""
        if base_model is None:
            base_model = self.config.get("model", {}).get("base_model", BASE_MODEL)
        cache_dir = self.config.get("model", {}).get("cache_dir", None)

        logger.info(f"Loading model: {base_model}")

        # Load processor with do_image_splitting=False as per TRL VLM example
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            do_image_splitting=False  # Required for VLM DPO training
        )

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
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
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
            num_proc=1,  # Single process to avoid pickling PIL images
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
                answer = item['answer']
                if isinstance(answer, list) and len(answer) > 0:
                    all_answers.append(answer[0])
                else:
                    all_answers.append(str(answer))
            elif 'label' in item:
                all_answers.append(str(item['label']))

        # Convert to DPO format
        dpo_data = []
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

            # Use fallback chain: no resize → 1920 → 1024 → 512
            image = prepare_image_with_fallback(image, f"benchmark_{benchmark_name}_{idx}")

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
                answer = item['answer']
                if isinstance(answer, list) and len(answer) > 0:
                    chosen = answer[0]
                else:
                    chosen = str(answer)
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

        # Log summary of skipped samples
        total_skipped = skipped_no_image + skipped_no_answer
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} samples from {benchmark_name}: "
                          f"{skipped_no_image} no image, {skipped_no_answer} no answer")

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

        logger.info(f"Training with DPO on benchmark: {benchmark_name}")

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
            dataset_num_proc=1,
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

        logger.info(f"Training with DPO on QCM: {dataset_path}")

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
            dataset_num_proc=1,
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

        logger.info(f"Training with DPO on: {dataset_path}")

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
            dataset_num_proc=1,
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
