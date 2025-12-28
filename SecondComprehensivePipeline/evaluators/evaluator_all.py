"""
Evaluator All - Runs all evaluations on a model
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json
import random
from datetime import datetime

from .evaluator_ocr import OCRBenchEvaluator
from .evaluator_docvqa import DocVQAEvaluator
from .evaluator_chartqa import ChartQAEvaluator
from .evaluator_qcm import QCMEvaluator
from .evaluator_qcm_claudette import QCMClaudetteEvaluator
from .evaluator_logprob import LogProbEvaluator
from .evaluator_bertscore import BertScoreEvaluator
from .evaluator_rouge import RougeEvaluator

logger = logging.getLogger(__name__)

# Fixed subset size and seed for consistent evaluation across runs
# This ensures we always evaluate on the same samples for fair comparison
EVAL_SUBSET_SIZE = 300  # For LogProb and ROUGE evaluations
BERTSCORE_SUBSET_SIZE = 50  # Smaller subset for BERTScore (slower evaluation)
DEBUG_SUBSET_SIZE = 2  # Default subset size for debug mode (overridden by config)
EVAL_SUBSET_SEED = 42


def get_fixed_subset_indices(dataset_size: int, subset_size: int = EVAL_SUBSET_SIZE, seed: int = EVAL_SUBSET_SEED) -> List[int]:
    """
    Get a fixed subset of indices for consistent evaluation.
    Uses a fixed seed to ensure the same samples are selected across all evaluations.

    Args:
        dataset_size: Total size of the dataset
        subset_size: Number of samples to select (default: 300)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of indices to use for evaluation
    """
    if dataset_size <= subset_size:
        return list(range(dataset_size))

    rng = random.Random(seed)
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    return sorted(indices[:subset_size])


class EvaluatorAll:
    """Runs all configured evaluations on a model"""

    def __init__(self, config: Dict[str, Any], cache_dir: str = None):
        self.config = config
        self.cache_dir = cache_dir
        self.results = {}

        # Initialize evaluators
        self.evaluators = {
            "ocrbench": OCRBenchEvaluator(cache_dir),
            "docvqa": DocVQAEvaluator(cache_dir),
            "chartqa": ChartQAEvaluator(cache_dir),
        }
        self.qcm_evaluator = QCMEvaluator(cache_dir)
        self.qcm_claudette_evaluator = QCMClaudetteEvaluator(cache_dir)
        self.logprob_evaluator = LogProbEvaluator(cache_dir)
        self.bertscore_evaluator = BertScoreEvaluator(cache_dir)
        self.rouge_evaluator = RougeEvaluator(cache_dir)

    def evaluate_all(self, model_path: str = None, model_name: str = "model", skip_bertscore: bool = True,
                     skip_datasets: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Run all enabled evaluations on the model

        Args:
            model_path: Path to model (None for base model)
            model_name: Name for this model in results
            skip_bertscore: If True, skip BERTScore evaluation (run it separately at the end)
            skip_datasets: Dict of dataset names to pre-computed results to skip re-evaluation.
                          e.g., {"qcm_gemini": {"accuracy": 85.2, "accuracy_train": 90.0, ...}}
                          Used when trainer has already computed accuracy on the training dataset.

        Returns:
            Dictionary with all evaluation results
        """
        skip_datasets = skip_datasets or {}
        logger.info(f"Running all evaluations for: {model_name}")
        start_time = datetime.now()

        # Check if debug mode is enabled (use debug_size samples for logprob/rouge/bertscore)
        debug_mode = self.config.get("pipeline", {}).get("debug_mode", False)
        debug_size = self.config.get("pipeline", {}).get("debug_size", DEBUG_SUBSET_SIZE)
        eval_subset_size = debug_size if debug_mode else EVAL_SUBSET_SIZE
        bertscore_subset_size = debug_size if debug_mode else BERTSCORE_SUBSET_SIZE
        if debug_mode:
            logger.info(f"DEBUG MODE - Using {debug_size} samples for logprob/rouge/bertscore evaluations")

        # Check if model path exists (if specified)
        if model_path:
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                logger.warning(f"Model path does not exist: {model_path}")
                logger.warning("Skipping evaluation for this model. Train first or use --eval-only for baseline only.")
                return {
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "error": f"Model path does not exist: {model_path}",
                    "benchmarks": {},
                    "erp_evaluation": {},
                    "summary": {}
                }

        all_results = {
            "model_name": model_name,
            "model_path": str(model_path) if model_path else "base_model",
            "timestamp": start_time.isoformat(),
            "benchmarks": {},
            "erp_evaluation": {},
            "summary": {}
        }

        # Evaluate on benchmarks
        eval_config = self.config.get("evaluation", {})
        benchmarks = eval_config.get("benchmarks", [])

        for benchmark in benchmarks:
            if not benchmark.get("enabled", True):
                continue

            name = benchmark["name"]
            max_samples = benchmark.get("max_samples", 1000)

            # Check if we have pre-computed results from training
            if name in skip_datasets:
                logger.info(f"Using pre-computed results for {name} (skipping re-evaluation)")
                all_results["benchmarks"][name] = skip_datasets[name]
                logger.info(f"{name}: {skip_datasets[name].get('accuracy', 'N/A')}% (pre-computed)")
                continue

            if name in self.evaluators:
                logger.info(f"Evaluating on {name}...")
                try:
                    evaluator = self.evaluators[name]
                    if model_path:
                        evaluator.load_model(model_path)
                    else:
                        evaluator.load_base_model()

                    result = evaluator.evaluate(max_samples=max_samples)
                    all_results["benchmarks"][name] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"]
                    }
                    logger.info(f"{name}: {result['accuracy']:.2f}%")
                except Exception as e:
                    logger.error(f"Error evaluating {name}: {e}")
                    all_results["benchmarks"][name] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after each evaluation
                    evaluator._cleanup_model()
        #*p - paste from system clipboard in normal mode
        # ERP-specific evaluations
        erp_config = eval_config.get("erp_evaluation", {})

        # QCM evaluations (Gemini, Nova, and Procedure)
        for qcm_name in ["qcm_gemini", "qcm_nova", "qcm_procedure1", "qcm_procedure2"]:
            qcm_config = erp_config.get(qcm_name, {})
            if qcm_config.get("enabled", False):
                # Check if we have pre-computed results from training
                if qcm_name in skip_datasets:
                    logger.info(f"Using pre-computed results for {qcm_name} (skipping re-evaluation)")
                    all_results["erp_evaluation"][qcm_name] = skip_datasets[qcm_name]
                    logger.info(f"ERP {qcm_name}: {skip_datasets[qcm_name].get('accuracy', 'N/A')}% (pre-computed)")
                    continue

                logger.info(f"Evaluating on ERP {qcm_name}...")
                try:
                    if model_path:
                        self.qcm_evaluator.load_model(model_path)
                    else:
                        self.qcm_evaluator.load_base_model()

                    # Resolve paths relative to SecondComprehensivePipeline folder
                    base_path = Path(__file__).parent.parent
                    dataset_path = base_path / qcm_config["dataset"]
                    image_dir = base_path / qcm_config["image_dir"]

                    result = self.qcm_evaluator.evaluate(
                        dataset_path=str(dataset_path),
                        image_dir=str(image_dir),
                        max_samples=qcm_config.get("max_samples")
                    )
                    all_results["erp_evaluation"][qcm_name] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "correct": result["correct"]
                    }
                    logger.info(f"ERP {qcm_name}: {result['accuracy']:.2f}%")
                except Exception as e:
                    logger.error(f"Error evaluating ERP {qcm_name}: {e}")
                    all_results["erp_evaluation"][qcm_name] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after each evaluation
                    self.qcm_evaluator._cleanup_model()

        # QCM Claudette evaluation (black images)
        qcm_claudette_config = erp_config.get("qcm_claudette", {})
        if qcm_claudette_config.get("enabled", False):
            # Check if we have pre-computed results from training
            if "qcm_claudette" in skip_datasets:
                logger.info("Using pre-computed results for qcm_claudette (skipping re-evaluation)")
                all_results["erp_evaluation"]["qcm_claudette"] = skip_datasets["qcm_claudette"]
                logger.info(f"QCM Claudette: {skip_datasets['qcm_claudette'].get('accuracy', 'N/A')}% (pre-computed)")
            else:
                logger.info("Evaluating on QCM Claudette (black images)...")
                try:
                    if model_path:
                        self.qcm_claudette_evaluator.load_model(model_path)
                    else:
                        self.qcm_claudette_evaluator.load_base_model()

                    # Resolve paths relative to SecondComprehensivePipeline folder
                    base_path = Path(__file__).parent.parent
                    dataset_path = base_path / qcm_claudette_config["dataset"]

                    result = self.qcm_claudette_evaluator.evaluate(
                        dataset_path=str(dataset_path),
                        max_samples=qcm_claudette_config.get("max_samples")
                    )
                    all_results["erp_evaluation"]["qcm_claudette"] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "correct": result["correct"]
                    }
                    logger.info(f"QCM Claudette: {result['accuracy']:.2f}%")
                except Exception as e:
                    logger.error(f"Error evaluating QCM Claudette: {e}")
                    all_results["erp_evaluation"]["qcm_claudette"] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after evaluation
                    self.qcm_claudette_evaluator._cleanup_model()

        # LogProb evaluations for Gemini and Nova datasets (separate)
        # Uses fixed 300-sample subset for consistent comparison across runs (10 in debug mode)
        for logprob_name in ["logprob_gemini", "logprob_nova"]:
            logprob_config = erp_config.get(logprob_name, {})
            if logprob_config.get("enabled", False):
                logger.info(f"Evaluating LogProb/Perplexity ({logprob_name}) with {eval_subset_size} samples...")
                try:
                    if model_path:
                        self.logprob_evaluator.load_model(model_path)
                    else:
                        self.logprob_evaluator.load_base_model()

                    base_path = Path(__file__).parent.parent
                    dataset_path = base_path / logprob_config["dataset"]
                    image_dir = base_path / logprob_config["image_dir"]

                    # Use fixed subset for consistent evaluation
                    result = self.logprob_evaluator.evaluate(
                        dataset_path=str(dataset_path),
                        image_dir=str(image_dir),
                        max_samples=eval_subset_size,
                        use_fixed_subset=True,
                        subset_seed=EVAL_SUBSET_SEED
                    )
                    all_results["erp_evaluation"][logprob_name] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "margin_mean": result["margin_mean"],
                        "chosen_avg_logprob": result.get("chosen_avg_logprob", 0.0),
                        "rejected_avg_logprob": result.get("rejected_avg_logprob", 0.0),
                        "chosen_perplexity": result.get("chosen_perplexity", 0.0),
                        "rejected_perplexity": result.get("rejected_perplexity", 0.0)
                    }
                    logger.info(f"{logprob_name} accuracy: {result['accuracy']:.2f}%, margin: {result['margin_mean']:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating {logprob_name}: {e}")
                    all_results["erp_evaluation"][logprob_name] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after evaluation
                    self.logprob_evaluator._cleanup_model()

        # BERTScore evaluations for Gemini and Nova datasets (separate)
        # Uses fixed 50-sample subset for consistent comparison (BERTScore is slow, 10 in debug mode)
        for bertscore_name in ["bertscore_gemini", "bertscore_nova"]:
            bertscore_config = erp_config.get(bertscore_name, {})
            if bertscore_config.get("enabled", False) and not skip_bertscore:
                logger.info(f"Evaluating with BERTScore ({bertscore_name}) with {bertscore_subset_size} samples...")
                try:
                    if model_path:
                        self.bertscore_evaluator.load_model(model_path)
                    else:
                        self.bertscore_evaluator.load_base_model()

                    base_path = Path(__file__).parent.parent
                    dataset_path = base_path / bertscore_config["dataset"]
                    image_dir = base_path / bertscore_config["image_dir"]

                    # Use fixed subset for consistent evaluation
                    result = self.bertscore_evaluator.evaluate(
                        dataset_path=str(dataset_path),
                        image_dir=str(image_dir),
                        max_samples=bertscore_subset_size,
                        lang=bertscore_config.get("lang", "en"),
                        use_fixed_subset=True,
                        subset_seed=EVAL_SUBSET_SEED
                    )
                    all_results["erp_evaluation"][bertscore_name] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "f1": result["f1"],
                        "precision": result["precision"],
                        "recall": result["recall"]
                    }
                    logger.info(f"{bertscore_name} F1: {result['f1']:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating {bertscore_name}: {e}")
                    all_results["erp_evaluation"][bertscore_name] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after evaluation
                    self.bertscore_evaluator._cleanup_model()
            elif bertscore_config.get("enabled", False) and skip_bertscore:
                logger.info(f"Skipping {bertscore_name} (will run at the end of pipeline)")

        # ROUGE evaluations for DPO datasets (Gemini and Nova)
        # Uses fixed 300-sample subset for consistent comparison across runs (10 in debug mode)
        for rouge_name in ["rouge_gemini", "rouge_nova"]:
            rouge_config = erp_config.get(rouge_name, {})
            if rouge_config.get("enabled", False):
                logger.info(f"Evaluating with ROUGE ({rouge_name}) with {eval_subset_size} samples...")
                try:
                    if model_path:
                        self.rouge_evaluator.load_model(model_path)
                    else:
                        self.rouge_evaluator.load_base_model()

                    base_path = Path(__file__).parent.parent
                    dataset_path = base_path / rouge_config["dataset"]
                    image_dir = base_path / rouge_config["image_dir"]

                    # Use fixed subset for consistent evaluation
                    result = self.rouge_evaluator.evaluate(
                        dataset_path=str(dataset_path),
                        image_dir=str(image_dir),
                        max_samples=eval_subset_size,
                        use_fixed_subset=True,
                        subset_seed=EVAL_SUBSET_SEED
                    )
                    all_results["erp_evaluation"][rouge_name] = {
                        "accuracy": result["accuracy"],
                        "total_samples": result["total_samples"],
                        "rouge1": result["rouge1"],
                        "rouge2": result["rouge2"],
                        "rougeL": result["rougeL"]
                    }
                    logger.info(f"{rouge_name} - ROUGE-1: {result['rouge1']:.4f}, ROUGE-2: {result['rouge2']:.4f}, ROUGE-L: {result['rougeL']:.4f}")
                except Exception as e:
                    logger.error(f"Error evaluating {rouge_name}: {e}")
                    all_results["erp_evaluation"][rouge_name] = {"error": str(e)}
                finally:
                    # Clean up GPU memory after evaluation
                    self.rouge_evaluator._cleanup_model()

        # Calculate summary
        benchmark_accs = [v["accuracy"] for v in all_results["benchmarks"].values() if "accuracy" in v]
        if benchmark_accs:
            all_results["summary"]["avg_benchmark_accuracy"] = sum(benchmark_accs) / len(benchmark_accs)

        # Add QCM accuracies to summary
        for qcm_name in ["qcm_gemini", "qcm_nova", "qcm_claudette", "qcm_procedure1", "qcm_procedure2"]:
            if qcm_name in all_results["erp_evaluation"] and "accuracy" in all_results["erp_evaluation"][qcm_name]:
                all_results["summary"][f"erp_{qcm_name}_accuracy"] = all_results["erp_evaluation"][qcm_name]["accuracy"]

        elapsed = datetime.now() - start_time
        all_results["summary"]["evaluation_time"] = str(elapsed)

        logger.info(f"All evaluations completed in {elapsed}")
        return all_results

    def evaluate_bertscore_only(self, model_path: str = None, model_name: str = "model") -> Dict[str, Any]:
        """
        Run only BERTScore evaluation on a model (for running at the end of pipeline)
        Evaluates both Gemini and Nova datasets separately with fixed 50-sample subsets.

        Args:
            model_path: Path to model (None for base model)
            model_name: Name for this model in results

        Returns:
            Dictionary with BERTScore results for both datasets
        """
        logger.info(f"Running BERTScore evaluation for: {model_name}")
        start_time = datetime.now()

        # Check if debug mode is enabled (use debug_size samples for bertscore)
        debug_mode = self.config.get("pipeline", {}).get("debug_mode", False)
        debug_size = self.config.get("pipeline", {}).get("debug_size", DEBUG_SUBSET_SIZE)
        bertscore_subset_size = debug_size if debug_mode else BERTSCORE_SUBSET_SIZE
        if debug_mode:
            logger.info(f"DEBUG MODE - Using {debug_size} samples for bertscore evaluation")

        eval_config = self.config.get("evaluation", {})
        erp_config = eval_config.get("erp_evaluation", {})

        result_data = {
            "model_name": model_name,
            "model_path": str(model_path) if model_path else "base_model",
        }

        # Evaluate both Gemini and Nova datasets
        for bertscore_name in ["bertscore_gemini", "bertscore_nova"]:
            bertscore_config = erp_config.get(bertscore_name, {})

            if not bertscore_config.get("enabled", False):
                logger.info(f"{bertscore_name} is disabled in config, skipping")
                result_data[bertscore_name] = {"skipped": True}
                continue

            logger.info(f"Evaluating {bertscore_name} with {bertscore_subset_size} samples...")
            try:
                if model_path:
                    self.bertscore_evaluator.load_model(model_path)
                else:
                    self.bertscore_evaluator.load_base_model()

                base_path = Path(__file__).parent.parent
                dataset_path = base_path / bertscore_config["dataset"]
                image_dir = base_path / bertscore_config["image_dir"]

                result = self.bertscore_evaluator.evaluate(
                    dataset_path=str(dataset_path),
                    image_dir=str(image_dir),
                    max_samples=bertscore_subset_size,
                    lang=bertscore_config.get("lang", "en"),
                    use_fixed_subset=True,
                    subset_seed=EVAL_SUBSET_SEED
                )
                result_data[bertscore_name] = {
                    "accuracy": result["accuracy"],
                    "total_samples": result["total_samples"],
                    "f1": result["f1"],
                    "precision": result["precision"],
                    "recall": result["recall"]
                }
                logger.info(f"{bertscore_name} F1: {result['f1']:.4f}")
            except Exception as e:
                logger.error(f"Error evaluating {bertscore_name}: {e}")
                result_data[bertscore_name] = {"error": str(e)}
            finally:
                # Clean up GPU memory after evaluation
                self.bertscore_evaluator._cleanup_model()

        elapsed = datetime.now() - start_time
        result_data["evaluation_time"] = str(elapsed)
        logger.info(f"BERTScore evaluation for {model_name} completed in {elapsed}")

        return result_data

    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save evaluation results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_name = results.get("model_name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.json"

        filepath = output_path / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {filepath}")
        return filepath


def evaluate_model(model_path: str = None, model_name: str = "model",
                   config: Dict[str, Any] = None, cache_dir: str = None,
                   output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model

    Args:
        model_path: Path to model (None for base model)
        model_name: Name for this model
        config: Configuration dictionary (from YAML)
        cache_dir: Cache directory for datasets
        output_dir: Output directory for results

    Returns:
        Evaluation results dictionary
    """
    evaluator = EvaluatorAll(config or {}, cache_dir)
    results = evaluator.evaluate_all(model_path, model_name)

    if output_dir:
        evaluator.save_results(results, output_dir)

    return results
