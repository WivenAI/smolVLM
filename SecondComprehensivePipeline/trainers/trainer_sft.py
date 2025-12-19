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

    def __init__(self, config: Dict[str, Any], output_dir: str, strategy_name: str,
                 processor=None, train_dataset=None, eval_dataset=None):
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

        # Reuse on_epoch_end logic with epoch=0
        self._run_evaluation(args, state, control, model, epoch=0, is_step_eval=True)

        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specific early steps to detect model breaking."""
        current_step = state.global_step

        # Check if this step should trigger evaluation
        if current_step in self.EARLY_EVAL_STEPS and current_step not in self._evaluated_steps:
            self._evaluated_steps.add(current_step)
            logger.info(f"[{self.strategy_name}] Running EARLY evaluation at step {current_step}...")
            self._run_evaluation(args, state, control, model, epoch=current_step, is_step_eval=True)

        return control

    def _compute_dataset_loss(self, model, dataset, dataset_name="dataset"):
        """Compute loss on a given dataset"""
        if dataset is None:
            return None

        model.eval()
        total_loss = 0.0
        num_batches = 0

        # Create a simple dataloader for the dataset
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

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
                    logger.warning(f"Error computing loss for batch in {dataset_name}: {e}")
                    continue

        model.train()

        if num_batches > 0:
            return total_loss / num_batches
        return None

    def _detect_dataset_type(self) -> str:
        """Detect dataset type from strategy name."""
        strategy_lower = self.strategy_name.lower()
        if 'qcm' in strategy_lower:
            return 'qcm'
        elif 'docvqa' in strategy_lower:
            return 'docvqa'
        elif 'ocr' in strategy_lower:
            return 'ocr'
        elif 'chart' in strategy_lower:
            return 'chartqa'
        elif 'dpo' in strategy_lower:
            return 'dpo'
        else:
            return 'benchmark'

    def _compute_dataset_accuracy(self, model, dataset, dataset_name="dataset"):
        """Compute accuracy on a dataset using the same metrics as the evaluators.

        Uses the appropriate evaluator's calculate_accuracy method based on dataset type.
        """
        if dataset is None or self.processor is None:
            return None, []

        dataset_type = self._detect_dataset_type()
        model.eval()
        results = []

        # Import evaluators to reuse their accuracy calculation logic
        from evaluators.evaluator_qcm import QCMEvaluator
        from evaluators.evaluator_docvqa import DocVQAEvaluator
        from evaluators.evaluator_ocr import OCRBenchEvaluator
        from evaluators.evaluator_chartqa import ChartQAEvaluator

        # Create a simple dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    labels = inputs.get('labels', None)
                    if labels is None:
                        continue

                    input_ids = inputs['input_ids']
                    attention_mask = inputs.get('attention_mask', None)
                    pixel_values = inputs.get('pixel_values', None)

                    gen_inputs = {'input_ids': input_ids}
                    if attention_mask is not None:
                        gen_inputs['attention_mask'] = attention_mask
                    if pixel_values is not None:
                        gen_inputs['pixel_values'] = pixel_values

                    # Generate more tokens for non-QCM datasets
                    max_tokens = 10 if dataset_type == 'qcm' else 50

                    outputs = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )

                    pred_tokens = outputs[0][input_ids.shape[1]:]
                    pred_text = self.processor.decode(pred_tokens, skip_special_tokens=True).strip()

                    label_tokens = labels[0][labels[0] != -100]
                    label_text = self.processor.decode(label_tokens, skip_special_tokens=True).strip()

                    # Format result for evaluator's calculate_accuracy method
                    results.append({
                        "response": pred_text,
                        "ground_truth": label_text,
                        "predicted_letter": pred_text.upper()[0] if pred_text else "",
                        "correct_answer": label_text.upper()[0] if label_text else "",
                        "is_correct": False  # Will be calculated by evaluator
                    })

                except Exception as e:
                    logger.warning(f"Error computing accuracy for batch {batch_idx} in {dataset_name}: {e}")
                    continue

        model.train()

        if not results:
            return None, results

        # Use the appropriate evaluator's calculate_accuracy method
        if dataset_type == 'qcm':
            # For QCM, set is_correct based on letter match
            for r in results:
                r['is_correct'] = r['predicted_letter'] == r['correct_answer']
            evaluator = QCMEvaluator()
            accuracy = evaluator.calculate_accuracy(results)
        elif dataset_type == 'docvqa':
            evaluator = DocVQAEvaluator()
            accuracy = evaluator.calculate_accuracy(results)
        elif dataset_type == 'ocr':
            evaluator = OCRBenchEvaluator()
            accuracy = evaluator.calculate_accuracy(results)
        elif dataset_type == 'chartqa':
            evaluator = ChartQAEvaluator()
            accuracy = evaluator.calculate_accuracy(results)
        else:
            # Default: use DocVQA-style matching (contains check)
            evaluator = DocVQAEvaluator()
            accuracy = evaluator.calculate_accuracy(results)

        logger.info(f"  [{dataset_name}] Accuracy ({dataset_type}): {accuracy:.2f}%")
        return accuracy, results

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
            train_loss = None
            train_accuracy = None
            train_results = []

            if not is_step_eval:
                train_loss = self._compute_dataset_loss(model, self.train_dataset, "train")
                train_accuracy, train_results = self._compute_dataset_accuracy(model, self.train_dataset, "train")

            test_loss = self._compute_dataset_loss(model, self.eval_dataset, "test")
            test_accuracy, test_results = self._compute_dataset_accuracy(model, self.eval_dataset, "test")

            # Log results
            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} Metrics:")
            if train_loss is not None:
                logger.info(f"  Train Loss: {train_loss:.4f}")
            if test_loss is not None:
                logger.info(f"  Test Loss: {test_loss:.4f}")
            if train_accuracy is not None:
                logger.info(f"  Train Accuracy: {train_accuracy:.2f}%")
            if test_accuracy is not None:
                logger.info(f"  Test Accuracy: {test_accuracy:.2f}%")

            # Check for memorization/overfitting (only for epoch eval)
            if not is_step_eval and train_accuracy is not None and test_accuracy is not None:
                gap = train_accuracy - test_accuracy
                if gap > 10:
                    logger.warning(f"  ⚠️ Large train-test gap ({gap:.2f}%): possible OVERFITTING")
                elif train_accuracy > 90 and test_accuracy > 80:
                    logger.info(f"  ✓ Good generalization (train: {train_accuracy:.1f}%, test: {test_accuracy:.1f}%)")
                elif train_accuracy < 50:
                    logger.warning(f"  ⚠️ Low train accuracy ({train_accuracy:.1f}%): possible UNDERFITTING")

            # Save the current model state
            model.save_pretrained(str(temp_model_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(temp_model_dir))

            # Import evaluator here to avoid circular imports
            from evaluators import EvaluatorAll

            # Run evaluation on standard benchmarks
            evaluator = EvaluatorAll(self.config, str(self.cache_dir))
            results = evaluator.evaluate_all(
                model_path=str(temp_model_dir),
                model_name=f"{self.strategy_name}_{eval_type}{epoch}"
            )

            # Log to WandB
            if WANDB_AVAILABLE and wandb.run is not None:
                metrics = {}

                # Log train/test losses
                if train_loss is not None:
                    metrics["eval/train_loss"] = train_loss
                if test_loss is not None:
                    metrics["eval/test_loss"] = test_loss

                # Log train/test accuracies
                if train_accuracy is not None:
                    metrics["eval/train_accuracy"] = train_accuracy
                if test_accuracy is not None:
                    metrics["eval/test_accuracy"] = test_accuracy

                # Log train-test gap (for memorization detection)
                if train_accuracy is not None and test_accuracy is not None:
                    metrics["eval/train_test_gap"] = train_accuracy - test_accuracy

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

                # Log average
                if results.get("summary", {}).get("avg_benchmark_accuracy"):
                    metrics["eval/avg_benchmark_acc"] = results["summary"]["avg_benchmark_accuracy"]

                # Log epoch/step number explicitly
                metrics[eval_type] = epoch
                metrics["global_step"] = state.global_step

                # Log all metrics at current step
                wandb.log(metrics, step=state.global_step)
                logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} eval metrics logged to WandB")

            # Log summary
            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} benchmark evaluation complete:")
            for key, value in results.get("summary", {}).items():
                if "accuracy" in key:
                    logger.info(f"  {key}: {value:.2f}%")

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Evaluation failed at {eval_type} {epoch}: {e}")
            import traceback
            traceback.print_exc()


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

        # Create evaluation callback with separate train/test datasets
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

        # Create evaluation callback with separate train/test datasets
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

        # Create evaluation callback with separate train/test datasets
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

        # Create evaluation callback with separate train/test datasets
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

        # Create evaluation callback with separate train/test datasets
        eval_callback = EpochEvaluationCallback(
            config=self.config,
            output_dir=output_dir,
            strategy_name=strategy_name or f"sft_{benchmark_name}",
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
