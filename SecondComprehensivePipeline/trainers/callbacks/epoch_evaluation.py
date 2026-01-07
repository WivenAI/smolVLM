"""
Epoch Evaluation Callbacks - Unified callbacks for all training strategies

Provides:
- BaseEpochEvaluationCallback: Shared functionality for all trainers
- SFTEpochEvaluationCallback: SFT-specific accuracy computation
- DPOEpochEvaluationCallback: DPO-specific preference metrics
- RAMMonitorCallback: System memory monitoring
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
from transformers import TrainerCallback

# Try to import optional dependencies
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from utils.dual_logger import log_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# RAM Monitor Callback
# =============================================================================

def get_ram_usage_gb() -> float:
    """Get current process RAM usage in GB"""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 3)
    return 0.0


def get_system_ram_info() -> Dict[str, float]:
    """Get system RAM info"""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024 ** 3),
            'available_gb': mem.available / (1024 ** 3),
            'used_gb': mem.used / (1024 ** 3),
            'percent': mem.percent
        }
    return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0}


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

            log_metrics({
                'system/process_ram_gb': current_ram,
                'system/ram_delta_gb': current_ram - self._initial_ram,
                'system/system_ram_used_gb': sys_ram['used_gb'],
                'system/system_ram_available_gb': sys_ram['available_gb'],
                'system/system_ram_percent': sys_ram['percent'],
            }, step=state.global_step)

        return control


# =============================================================================
# Base Epoch Evaluation Callback
# =============================================================================

class BaseEpochEvaluationCallback(TrainerCallback, ABC):
    """
    Base callback with shared evaluation logic for all training strategies.

    Shared functionality:
    - Evaluation scheduling (at specific steps and epochs)
    - Baseline results logging at epoch 0
    - TensorBoard and WandB integration
    - Model saving during evaluation
    - EvaluatorAll integration for benchmark evaluation

    Subclasses must implement:
    - _compute_metrics(): Compute task-specific metrics (accuracy or preference)
    - _log_task_metrics(): Log task-specific metrics to tracking systems
    """

    # Steps at which to run early evaluation (to detect model breaking)
    EARLY_EVAL_STEPS = [200]

    # Epochs at which to run evaluation
    EVAL_EPOCHS = [1, 2, 5, 10, 20, 30, 40, 100]

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str,
        strategy_name: str,
        processor,
        train_dataset=None,
        eval_dataset=None,
        training_dataset_name: Optional[str] = None
    ):
        """
        Initialize the callback.

        Args:
            config: Training configuration dictionary
            output_dir: Directory to save model checkpoints during evaluation
            strategy_name: Name of the training strategy (for logging)
            processor: Model processor for tokenization
            train_dataset: Training dataset for train metrics
            eval_dataset: Evaluation dataset for test metrics
            training_dataset_name: Optional name of training dataset (for EvaluatorAll skip)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.strategy_name = strategy_name
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_dataset_name = training_dataset_name
        self.cache_dir = Path(__file__).parent.parent.parent / "datasets" / "cache"
        self._initial_eval_done = False
        self._evaluated_steps = set()

        # Setup TensorBoard logging
        self.tensorboard_writer = None
        if TENSORBOARD_AVAILABLE:
            tensorboard_dir = Path(__file__).parent.parent.parent / "tensorboard_logs" / strategy_name
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")

        # Setup WandB with offline fallback
        self.wandb_enabled = False
        if WANDB_AVAILABLE and self.config.get("pipeline", {}).get("use_wandb", True):
            try:
                if wandb.run is not None:
                    self.wandb_enabled = True
                    logger.info(f"WandB logging enabled (online mode)")
                else:
                    os.environ["WANDB_MODE"] = "offline"
                    self.wandb_enabled = True
                    logger.info(f"WandB logging enabled (offline mode)")
            except Exception as e:
                logger.warning(f"WandB initialization failed, continuing without WandB: {e}")
                self.wandb_enabled = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Load baseline results and log them at epoch 0 for comparison graphs."""
        logger.info(f"[{self.strategy_name}] Loading baseline results for epoch 0 logging...")

        try:
            results_dir = Path(__file__).parent.parent.parent / "results"
            from evaluators.evaluator_all import EvaluatorAll
            baseline_results = EvaluatorAll.load_baseline_results(str(results_dir))

            if baseline_results:
                self._log_baseline_results_at_epoch_0(baseline_results, state)
                logger.info(f"[{self.strategy_name}] Baseline results logged at epoch 0")
            else:
                logger.warning(f"[{self.strategy_name}] No baseline results found - run baseline evaluation first")

        except Exception as e:
            logger.warning(f"[{self.strategy_name}] Failed to load baseline results: {e}")

        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specific early steps to detect model breaking."""
        current_step = state.global_step

        if current_step in self.EARLY_EVAL_STEPS and current_step not in self._evaluated_steps:
            self._evaluated_steps.add(current_step)
            logger.info(f"[{self.strategy_name}] Running FULL evaluation at step {current_step}...")
            self._run_evaluation(args, state, control, model, epoch=current_step, is_step_eval=False)

        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Run full evaluation at end of specific epochs on both train and test sets."""
        epoch = int(state.epoch)
        if epoch not in self.EVAL_EPOCHS:
            logger.info(f"[{self.strategy_name}] Skipping evaluation at epoch {epoch} (not in EVAL_EPOCHS)")
            return control
        logger.info(f"[{self.strategy_name}] Running train/test evaluation at epoch {epoch}...")
        self._run_evaluation(args, state, control, model, epoch=epoch, is_step_eval=False)
        return control

    def _log_baseline_results_at_epoch_0(self, baseline_results: Dict[str, Any], state):
        """
        Log baseline results at epoch 0 for comparison graphs.
        Logs to both WandB and TensorBoard for redundancy.
        """
        try:
            metrics = {}

            # Log benchmark accuracies and ANLS at epoch 0
            for bench_name, bench_data in baseline_results.get("benchmarks", {}).items():
                if "accuracy" in bench_data:
                    metrics[f"eval/{bench_name}_acc"] = bench_data["accuracy"]
                if "anls" in bench_data:
                    metrics[f"eval/{bench_name}_anls"] = bench_data["anls"]

            # Log ERP evaluation metrics at epoch 0
            erp = baseline_results.get("erp_evaluation", {})

            # QCM metrics
            for qcm_name in ["qcm_gemini", "qcm_nova", "qcm_claudette", "qcm_procedure1", "qcm_procedure2"]:
                if qcm_name in erp and "accuracy" in erp[qcm_name]:
                    metrics[f"eval/{qcm_name}_acc"] = erp[qcm_name]["accuracy"]

            # LogProb/Perplexity metrics
            for logprob_name in ["logprob_gemini", "logprob_nova"]:
                if logprob_name in erp and "accuracy" in erp[logprob_name]:
                    metrics[f"eval/{logprob_name}_acc"] = erp[logprob_name]["accuracy"]
                    metrics[f"eval/{logprob_name}_margin"] = erp[logprob_name].get("margin_mean", 0)
                    metrics[f"eval/{logprob_name}_chosen_ppl"] = erp[logprob_name].get("chosen_perplexity", 0)
                    metrics[f"eval/{logprob_name}_rejected_ppl"] = erp[logprob_name].get("rejected_perplexity", 0)

            # ROUGE metrics
            for rouge_name in ["rouge_gemini", "rouge_nova"]:
                if rouge_name in erp and "accuracy" in erp[rouge_name]:
                    metrics[f"eval/{rouge_name}_acc"] = erp[rouge_name]["accuracy"]
                    metrics[f"eval/{rouge_name}_rouge1"] = erp[rouge_name].get("rouge1", 0)
                    metrics[f"eval/{rouge_name}_rouge2"] = erp[rouge_name].get("rouge2", 0)
                    metrics[f"eval/{rouge_name}_rougeL"] = erp[rouge_name].get("rougeL", 0)

            # BERTScore metrics
            for bertscore_name in ["bertscore_gemini", "bertscore_nova"]:
                if bertscore_name in erp and "f1" in erp[bertscore_name]:
                    metrics[f"eval/{bertscore_name}_f1"] = erp[bertscore_name]["f1"]
                    metrics[f"eval/{bertscore_name}_precision"] = erp[bertscore_name].get("precision", 0)
                    metrics[f"eval/{bertscore_name}_recall"] = erp[bertscore_name].get("recall", 0)

            # Log average benchmark accuracy
            if baseline_results.get("summary", {}).get("avg_benchmark_accuracy"):
                metrics["eval/avg_benchmark_acc"] = baseline_results["summary"]["avg_benchmark_accuracy"]

            metrics["epoch"] = 0
            metrics["global_step"] = 0

            # Log to both WandB and TensorBoard
            log_metrics(metrics, step=0)
            logger.info(f"[{self.strategy_name}] Baseline metrics logged at epoch 0")

            # Legacy TensorBoard writer
            if self.tensorboard_writer:
                try:
                    for key, value in metrics.items():
                        if key not in ["epoch", "global_step"] and isinstance(value, (int, float)):
                            self.tensorboard_writer.add_scalar(key, value, global_step=0)
                    self.tensorboard_writer.flush()
                    logger.info(f"[{self.strategy_name}] Baseline metrics logged to TensorBoard at epoch 0")
                except Exception as e:
                    logger.warning(f"[{self.strategy_name}] TensorBoard logging failed: {e}")

        except Exception as e:
            logger.warning(f"[{self.strategy_name}] Failed to log baseline results: {e}")
            import traceback
            traceback.print_exc()

    def _compute_dataset_loss(self, model, dataset, dataset_name: str = "dataset") -> Optional[float]:
        """Compute loss on a given dataset."""
        if dataset is None or model is None:
            return None

        model.eval()
        total_loss = 0.0
        num_batches = 0

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in dataloader:
                try:
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

    @abstractmethod
    def _compute_metrics(
        self,
        model,
        dataset,
        dataset_name: str
    ) -> Tuple[Optional[float], Any]:
        """
        Compute task-specific metrics.

        Must be implemented by subclasses.

        Returns:
            Tuple of (primary_metric, additional_data)
            - For SFT: (accuracy, results_list)
            - For DPO: (preference_accuracy, margin)
        """
        pass

    @abstractmethod
    def _log_task_metrics(
        self,
        metrics: Dict[str, Any],
        train_result: Any,
        test_result: Any,
        epoch: int,
        is_step_eval: bool
    ) -> None:
        """
        Log task-specific metrics.

        Must be implemented by subclasses.
        """
        pass

    def _run_evaluation(self, args, state, control, model, epoch: int, is_step_eval: bool = False):
        """
        Shared evaluation logic for both step-based and epoch-based evaluation.
        """
        eval_type = "step" if is_step_eval else "epoch"
        temp_model_dir = self.output_dir / f"{eval_type}_{epoch}_eval"
        temp_model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Compute metrics
            train_result = (None, None)
            if not is_step_eval:
                train_result = self._compute_metrics(model, self.train_dataset, "train")

            test_result = self._compute_metrics(model, self.eval_dataset, "test")

            # Save model state
            model.save_pretrained(str(temp_model_dir))
            if self.processor is not None:
                self.processor.save_pretrained(str(temp_model_dir))

            # Run EvaluatorAll
            from evaluators import EvaluatorAll

            skip_datasets = self._build_skip_datasets(train_result, test_result, is_step_eval)

            evaluator = EvaluatorAll(self.config, str(self.cache_dir))
            results = evaluator.evaluate_all(
                model_path=str(temp_model_dir),
                model_name=f"{self.strategy_name}_{eval_type}{epoch}",
                skip_datasets=skip_datasets
            )

            # Collect metrics
            metrics = {}

            # Let subclass add task-specific metrics
            self._log_task_metrics(metrics, train_result, test_result, epoch, is_step_eval)

            # Log benchmark accuracies and ANLS
            for bench_name, bench_data in results.get("benchmarks", {}).items():
                if "accuracy" in bench_data:
                    metrics[f"eval/{bench_name}_acc"] = bench_data["accuracy"]
                if "anls" in bench_data:
                    metrics[f"eval/{bench_name}_anls"] = bench_data["anls"]
                if "skipped_samples" in bench_data:
                    metrics[f"eval/{bench_name}_skipped"] = bench_data["skipped_samples"]

            # Log ERP evaluation metrics
            self._log_erp_metrics(metrics, results.get("erp_evaluation", {}))

            # Log average
            if results.get("summary", {}).get("avg_benchmark_accuracy"):
                metrics["eval/avg_benchmark_acc"] = results["summary"]["avg_benchmark_accuracy"]

            metrics[eval_type] = epoch
            metrics["global_step"] = state.global_step

            # Log to both WandB and TensorBoard
            log_metrics(metrics, step=state.global_step)
            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} eval metrics logged")

            # Legacy TensorBoard writer
            if self.tensorboard_writer:
                try:
                    for key, value in metrics.items():
                        if key not in ["epoch", "global_step", "eval/split_type"] and isinstance(value, (int, float)):
                            self.tensorboard_writer.add_scalar(key, value, global_step=state.global_step)
                    self.tensorboard_writer.flush()
                except Exception as e:
                    logger.warning(f"[{self.strategy_name}] TensorBoard logging failed: {e}")

            # Log summary
            logger.info(f"[{self.strategy_name}] {eval_type.capitalize()} {epoch} benchmark evaluation complete:")
            for key, value in results.get("summary", {}).items():
                if "accuracy" in key:
                    logger.info(f"  {key}: {value:.2f}%")

        except Exception as e:
            logger.error(f"[{self.strategy_name}] Evaluation failed at {eval_type} {epoch}: {e}")
            import traceback
            traceback.print_exc()

    def _build_skip_datasets(self, train_result, test_result, is_step_eval: bool) -> Dict[str, Any]:
        """Build skip_datasets dict for EvaluatorAll. Override in subclass if needed."""
        return {}

    def _log_erp_metrics(self, metrics: Dict[str, Any], erp: Dict[str, Any]) -> None:
        """Log ERP evaluation metrics."""
        # QCM metrics
        for qcm_name in ["qcm_gemini", "qcm_nova", "qcm_claudette", "qcm_procedure1", "qcm_procedure2"]:
            if qcm_name in erp and "accuracy" in erp[qcm_name]:
                metrics[f"eval/{qcm_name}_acc"] = erp[qcm_name]["accuracy"]

        # LogProb metrics
        for logprob_name in ["logprob_gemini", "logprob_nova"]:
            if logprob_name in erp and "accuracy" in erp[logprob_name]:
                metrics[f"eval/{logprob_name}_acc"] = erp[logprob_name]["accuracy"]
                metrics[f"eval/{logprob_name}_margin"] = erp[logprob_name].get("margin_mean", 0)
                metrics[f"eval/{logprob_name}_chosen_ppl"] = erp[logprob_name].get("chosen_perplexity", 0)
                metrics[f"eval/{logprob_name}_rejected_ppl"] = erp[logprob_name].get("rejected_perplexity", 0)
                if "skipped_samples" in erp[logprob_name]:
                    metrics[f"eval/{logprob_name}_skipped"] = erp[logprob_name]["skipped_samples"]

        # ROUGE metrics
        for rouge_name in ["rouge_gemini", "rouge_nova"]:
            if rouge_name in erp and "accuracy" in erp[rouge_name]:
                metrics[f"eval/{rouge_name}_acc"] = erp[rouge_name]["accuracy"]
                metrics[f"eval/{rouge_name}_rouge1"] = erp[rouge_name].get("rouge1", 0)
                metrics[f"eval/{rouge_name}_rouge2"] = erp[rouge_name].get("rouge2", 0)
                metrics[f"eval/{rouge_name}_rougeL"] = erp[rouge_name].get("rougeL", 0)
                if "skipped_samples" in erp[rouge_name]:
                    metrics[f"eval/{rouge_name}_skipped"] = erp[rouge_name]["skipped_samples"]

        # BERTScore metrics
        for bertscore_name in ["bertscore_gemini", "bertscore_nova"]:
            if bertscore_name in erp and "f1" in erp[bertscore_name]:
                metrics[f"eval/{bertscore_name}_f1"] = erp[bertscore_name]["f1"]
                metrics[f"eval/{bertscore_name}_precision"] = erp[bertscore_name].get("precision", 0)
                metrics[f"eval/{bertscore_name}_recall"] = erp[bertscore_name].get("recall", 0)
                if "skipped_samples" in erp[bertscore_name]:
                    metrics[f"eval/{bertscore_name}_skipped"] = erp[bertscore_name]["skipped_samples"]


# =============================================================================
# SFT Epoch Evaluation Callback
# =============================================================================

class SFTEpochEvaluationCallback(BaseEpochEvaluationCallback):
    """
    SFT-specific evaluation callback.

    Computes accuracy metrics for QCM and benchmark datasets.
    """

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

    def _compute_metrics(
        self,
        model,
        dataset,
        dataset_name: str
    ) -> Tuple[Optional[float], List[Dict]]:
        """Compute accuracy metrics for SFT training."""
        return self._compute_dataset_accuracy(model, dataset, dataset_name)

    def _compute_dataset_accuracy(self, model, dataset, dataset_name: str = "dataset") -> Tuple[Optional[float], List[Dict]]:
        """
        Compute accuracy on a dataset using the shared QCM accuracy module.
        """
        if dataset is None or self.processor is None or model is None:
            return None, []

        dataset_type = self._detect_dataset_type()
        model.eval()
        results = []

        from evaluators.qcm_accuracy import calculate_qcm_accuracy, extract_answer_letter, normalize_text
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

                    # Find where answer starts
                    prompt_end_pos = None
                    if labels is not None:
                        mask = labels[0] != -100
                        if mask.any():
                            prompt_end_pos = mask.nonzero()[0].item()

                    if prompt_end_pos is None or prompt_end_pos >= input_ids.shape[1]:
                        prompt_end_pos = input_ids.shape[1]
                        logger.warning(f"Could not find answer position for batch {batch_idx}, using full sequence")

                    # Trim to prompt only
                    prompt_input_ids = input_ids[:, :prompt_end_pos]
                    prompt_attention_mask = attention_mask[:, :prompt_end_pos] if attention_mask is not None else None

                    gen_inputs = {'input_ids': prompt_input_ids}
                    if prompt_attention_mask is not None:
                        gen_inputs['attention_mask'] = prompt_attention_mask
                    if pixel_values is not None:
                        gen_inputs['pixel_values'] = pixel_values

                    max_tokens = 10 if dataset_type == 'qcm' else 50

                    outputs = model.generate(
                        **gen_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )

                    pred_tokens = outputs[0][prompt_input_ids.shape[1]:]
                    pred_text = self.processor.decode(pred_tokens, skip_special_tokens=True).strip()

                    label_tokens = labels[0][labels[0] != -100]
                    label_text = self.processor.decode(label_tokens, skip_special_tokens=True).strip()

                    if dataset_type == 'qcm':
                        predicted_letter = extract_answer_letter(pred_text, ['A', 'B', 'C', 'D'])
                    else:
                        predicted_letter = pred_text.upper()[0] if pred_text else ""

                    results.append({
                        "response": pred_text,
                        "ground_truth": label_text,
                        "predicted_letter": predicted_letter,
                        "correct_answer": label_text.upper()[0] if label_text else "",
                        "is_correct": False
                    })

                except Exception as e:
                    logger.warning(f"Error computing accuracy for batch {batch_idx} in {dataset_name}: {e}")
                    continue

        model.train()

        if not results:
            return None, results

        split = "train" if "train" in dataset_name.lower() else "test"

        import wandb
        log_to_wandb = wandb.run is not None

        if dataset_type == 'qcm':
            metrics = calculate_qcm_accuracy(
                results,
                split=split,
                log_to_wandb=log_to_wandb,
                wandb_prefix=self.strategy_name
            )
            accuracy = metrics["lenient_accuracy"]
            logger.info(f"  [{dataset_name}] Lenient: {accuracy:.2f}% ({dataset_type}, {split})")
        else:
            correct = sum(1 for r in results if normalize_text(r['response']) == normalize_text(r['ground_truth']))
            total = len(results)
            accuracy = (correct / total * 100) if total > 0 else 0.0

            if log_to_wandb:
                log_metrics({
                    f"{self.strategy_name}/{split}_accuracy": accuracy,
                    f"{self.strategy_name}/{split}_correct": correct,
                    f"{self.strategy_name}/{split}_total": total
                })

            logger.info(f"  [{dataset_name}] Accuracy: {accuracy:.2f}% ({dataset_type}, {split})")

        return accuracy, results

    def _log_task_metrics(
        self,
        metrics: Dict[str, Any],
        train_result: Tuple[Optional[float], List[Dict]],
        test_result: Tuple[Optional[float], List[Dict]],
        epoch: int,
        is_step_eval: bool
    ) -> None:
        """Log SFT-specific accuracy metrics."""
        train_accuracy, train_results = train_result
        test_accuracy, test_results = test_result

        # Note: Losses are already computed and logged in _run_evaluation
        # We only log accuracy metrics here

        # Log accuracies
        if train_accuracy is not None:
            metrics["eval/accuracy_train"] = train_accuracy
            logger.info(f"  Train Accuracy: {train_accuracy:.2f}%")
            print(f">>> ACCURACY [{self.strategy_name}] Epoch {epoch} TRAIN: {train_accuracy:.2f}%", flush=True)

        if test_accuracy is not None:
            metrics["eval/accuracy_test"] = test_accuracy
            logger.info(f"  Test Accuracy: {test_accuracy:.2f}%")
            print(f">>> ACCURACY [{self.strategy_name}] Epoch {epoch} TEST: {test_accuracy:.2f}%", flush=True)

        # Calculate full accuracy
        if train_accuracy is not None and test_accuracy is not None:
            train_size = len(train_results) if train_results else 0
            test_size = len(test_results) if test_results else 0
            total_size = train_size + test_size
            if total_size > 0:
                full_accuracy = (train_accuracy * train_size + test_accuracy * test_size) / total_size
                metrics["eval/accuracy_full"] = full_accuracy

            metrics["eval/train_test_gap"] = train_accuracy - test_accuracy

            # Check for overfitting
            gap = train_accuracy - test_accuracy
            if gap > 10:
                logger.warning(f"  ⚠️ Large train-test gap ({gap:.2f}%): possible OVERFITTING")
            elif train_accuracy > 90 and test_accuracy > 80:
                logger.info(f"  ✓ Good generalization (train: {train_accuracy:.1f}%, test: {test_accuracy:.1f}%)")
            elif train_accuracy < 50:
                logger.warning(f"  ⚠️ Low train accuracy ({train_accuracy:.1f}%): possible UNDERFITTING")

        metrics["eval/split_type"] = 0 if is_step_eval else 1

    def _build_skip_datasets(self, train_result, test_result, is_step_eval: bool) -> Dict[str, Any]:
        """Build skip_datasets dict with pre-computed results."""
        skip_datasets = {}

        if self.training_dataset_name and not is_step_eval:
            train_accuracy, train_results = train_result
            test_accuracy, test_results = test_result

            train_size = len(train_results) if train_results else 0
            test_size = len(test_results) if test_results else 0
            total_size = train_size + test_size

            full_accuracy = None
            if total_size > 0 and train_accuracy is not None and test_accuracy is not None:
                full_accuracy = (train_accuracy * train_size + test_accuracy * test_size) / total_size

            skip_datasets[self.training_dataset_name] = {
                "accuracy": full_accuracy if full_accuracy is not None else test_accuracy,
                "accuracy_train": train_accuracy,
                "accuracy_test": test_accuracy,
                "total_samples": total_size,
                "correct": sum(1 for r in (train_results + test_results) if r.get('is_correct', False)),
                "pre_computed": True
            }
            logger.info(f"Passing pre-computed results for {self.training_dataset_name} to EvaluatorAll")

        return skip_datasets


# =============================================================================
# DPO Epoch Evaluation Callback
# =============================================================================

class DPOEpochEvaluationCallback(BaseEpochEvaluationCallback):
    """
    DPO-specific evaluation callback.

    Computes preference accuracy (chosen > rejected) and margin metrics.
    """

    def _compute_metrics(
        self,
        model,
        dataset,
        dataset_name: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute DPO preference metrics."""
        return self._compute_dpo_metrics(model, dataset, dataset_name)

    def _compute_dpo_metrics(self, model, dataset, dataset_name: str = "dataset") -> Tuple[Optional[float], Optional[float]]:
        """
        Compute DPO-specific metrics: preference accuracy (chosen > rejected).

        Returns:
            Tuple of (preference_accuracy, avg_margin)
        """
        if dataset is None or self.processor is None or model is None:
            return None, None

        model.eval()
        correct_preferences = 0
        total = 0
        total_margin = 0.0

        with torch.no_grad():
            for idx in range(len(dataset)):
                try:
                    item = dataset[idx]

                    prompt = item.get('prompt', None)
                    chosen = item.get('chosen', None)
                    rejected = item.get('rejected', None)
                    images = item.get('images', None)

                    # Strict validation - DPO requires all fields and images
                    if not prompt:
                        raise ValueError(f"DPO sample {idx} in {dataset_name} missing prompt")
                    if not chosen:
                        raise ValueError(f"DPO sample {idx} in {dataset_name} missing chosen response")
                    if not rejected:
                        raise ValueError(f"DPO sample {idx} in {dataset_name} missing rejected response")
                    if not images or len(images) == 0:
                        raise ValueError(f"DPO sample {idx} in {dataset_name} missing images - "
                                       f"VLM DPO requires images for all samples")

                    device = next(model.parameters()).device
                    image = images[0]

                    # Handle chat format (list of message dicts) vs plain text
                    if isinstance(prompt, list):
                        # Chat format - use processor's apply_chat_template
                        chosen_messages = prompt + chosen
                        rejected_messages = prompt + rejected
                        chosen_text = self.processor.apply_chat_template(chosen_messages, tokenize=False)
                        rejected_text = self.processor.apply_chat_template(rejected_messages, tokenize=False)
                    else:
                        # Plain text format
                        chosen_text = f"{prompt}{chosen}"
                        rejected_text = f"{prompt}{rejected}"

                    # Compute log probs for chosen
                    chosen_inputs = self.processor(
                        text=chosen_text,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )
                    chosen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in chosen_inputs.items()}
                    chosen_inputs['labels'] = chosen_inputs['input_ids'].clone()
                    chosen_outputs = model(**chosen_inputs)
                    chosen_loss = chosen_outputs.loss.item()

                    # Compute log probs for rejected
                    rejected_inputs = self.processor(
                        text=rejected_text,
                        images=image,
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

    def _log_task_metrics(
        self,
        metrics: Dict[str, Any],
        train_result: Tuple[Optional[float], Optional[float]],
        test_result: Tuple[Optional[float], Optional[float]],
        epoch: int,
        is_step_eval: bool
    ) -> None:
        """Log DPO-specific preference metrics."""
        train_pref_acc, train_margin = train_result
        test_pref_acc, test_margin = test_result

        logger.info(f"[{self.strategy_name}] Epoch {epoch} DPO Metrics:")

        if train_pref_acc is not None:
            metrics["eval/train_preference_acc"] = train_pref_acc
            metrics["eval/train_margin"] = train_margin
            logger.info(f"  Train Preference Accuracy: {train_pref_acc:.2f}% (margin: {train_margin:.4f})")

        if test_pref_acc is not None:
            metrics["eval/test_preference_acc"] = test_pref_acc
            metrics["eval/test_margin"] = test_margin
            logger.info(f"  Test Preference Accuracy: {test_pref_acc:.2f}% (margin: {test_margin:.4f})")

        # Log train-test gap
        if train_pref_acc is not None and test_pref_acc is not None:
            metrics["eval/train_test_gap"] = train_pref_acc - test_pref_acc

            # Check for overfitting
            gap = train_pref_acc - test_pref_acc
            if gap > 10:
                logger.warning(f"  ⚠️ Large train-test gap ({gap:.2f}%): possible OVERFITTING")
            elif train_pref_acc > 90 and test_pref_acc > 80:
                logger.info(f"  ✓ Good generalization (train: {train_pref_acc:.1f}%, test: {test_pref_acc:.1f}%)")
            elif train_pref_acc < 60:
                logger.warning(f"  ⚠️ Low train preference accuracy ({train_pref_acc:.1f}%): DPO not learning preferences")
