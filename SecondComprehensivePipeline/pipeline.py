#!/usr/bin/env python3
"""
Pipeline - Clean, modular training and evaluation pipeline

This pipeline:
1. Reads configuration from config/conf.yaml
2. Trains models according to enabled strategies
3. Evaluates all models on all benchmarks
4. Saves results and generates comparison reports

Usage:
    python pipeline.py                    # Run full pipeline
    python pipeline.py --eval-only        # Evaluation only
    python pipeline.py --debug            # Debug mode (10 samples)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import logging
import yaml
import random
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeeOutput:
    """Capture output to both stdout and file"""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: str = None):
        self.base_path = Path(__file__).parent
        self.config_path = Path(config_path) if config_path else self.base_path / "config" / "conf.yaml"
        self.config = self._load_config()

        # Setup directories
        self.paths = self.config.get("paths", {})
        self.output_dir = self.base_path / self.paths.get("output_dir", "modelweights")
        self.cache_dir = self.base_path / self.paths.get("cache_dir", "datasets/cache")
        self.results_dir = self.base_path / self.paths.get("results_dir", "results")
        self.logs_dir = self.base_path / self.paths.get("logs_dir", "logs")

        for d in [self.output_dir, self.cache_dir, self.results_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging
        self.log_file = self.logs_dir / f"pipeline_{self.timestamp}.log"
        self.tee = TeeOutput(self.log_file)
        sys.stdout = self.tee

        # Track results
        self.all_results = {}

        logger.info(f"Pipeline initialized")
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Log: {self.log_file}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Convert relative cache_dir to absolute path (relative to project root)
        if "model" in config and "cache_dir" in config["model"]:
            cache_dir = config["model"]["cache_dir"]
            if not os.path.isabs(cache_dir):
                config["model"]["cache_dir"] = str(self.base_path / cache_dir)
                logger.info(f"Using model.cache_dir: {config['model']['cache_dir']}")

        logger.info(f"Loaded config from: {self.config_path}")
        return config

    def _apply_debug_mode(self):
        """Apply debug mode settings"""
        if self.config.get("pipeline", {}).get("debug_mode", False):
            debug_size = self.config.get("pipeline", {}).get("debug_size", 1)
            logger.info(f"DEBUG MODE - Using {debug_size} samples for everything")
            self.config["training"]["train_samples"] = debug_size
            self.config["training"]["epochs"] = 1
            self.config["pipeline"]["use_wandb"] = False  # Disable wandb in debug mode
            for bench in self.config.get("evaluation", {}).get("benchmarks", []):
                bench["max_samples"] = debug_size
            # Apply to ERP evaluations too
            for erp_eval in self.config.get("evaluation", {}).get("erp_evaluation", {}).values():
                if isinstance(erp_eval, dict) and "max_samples" in erp_eval:
                    erp_eval["max_samples"] = debug_size
            # Use separate folders for debug mode
            self.output_dir = self.base_path / "modelweights_debug"
            self.results_dir = self.base_path / "results_debug"
            self.cache_dir = self.base_path / "tmpcache_debug"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Update config cache_dir for debug mode
            self.config["model"]["cache_dir"] = str(self.cache_dir)
            logger.info(f"DEBUG MODE - Using folders: {self.output_dir}, {self.results_dir}, {self.cache_dir}")

    def run(self, eval_only: bool = False, compare_only: bool = False) -> Dict[str, Any]:
        """Run the full pipeline"""
        self._apply_debug_mode()

        start_time = datetime.now()

        print("=" * 80)
        print("NoHallucinations Pipeline")
        print("=" * 80)
        print(f"Started: {start_time}")
        print(f"Config: {self.config_path}")
        print("=" * 80)

        # Get enabled strategies
        strategies = self.config.get("training", {}).get("strategies", [])
        enabled_strategies = [s for s in strategies if s.get("enabled", True)]

        if not compare_only:
            print(f"\nEnabled training strategies: {len(enabled_strategies)}")
            for s in enabled_strategies:
                print(f"  - {s['name']} ({s['type']})")

        # Phase 1: Training (if not eval-only or compare-only)
        if not eval_only and not compare_only:
            self._run_training_phase(enabled_strategies)

        # Phase 2: Evaluation (if not compare-only)
        if not compare_only:
            self._run_evaluation_phase(enabled_strategies)
        else:
            # Load existing results from result files
            self._load_existing_results()

        # Phase 3: Sample Outputs
        if not compare_only:
            self._save_sample_outputs(enabled_strategies)

        # Phase 4: Comparison
        self._generate_comparison()

        # Phase 5: BERTScore evaluation (runs last as it's slow)
        if not compare_only:
            self._run_bertscore_phase(enabled_strategies)

        # Summary
        elapsed = datetime.now() - start_time
        print("\n" + "=" * 80)
        print("Pipeline completed!")
        print("=" * 80)
        print(f"Total time: {elapsed}")
        print(f"Results: {self.results_dir}")
        print(f"Log: {self.log_file}")
        print("=" * 80)

        # Cleanup
        sys.stdout = self.tee.terminal
        self.tee.close()

        return self.all_results

    def _run_training_phase(self, strategies: List[Dict]):
        """Run training for all enabled strategies"""
        print("\n" + "=" * 80)
        print("PHASE 1: TRAINING")
        print("=" * 80)

        from trainers.trainer_sft import train_sft
        from trainers.trainer_dpo import train_dpo
        from trainers.trainer_full_finetune import train_full_finetune

        for strategy in strategies:
            name = strategy["name"]
            strategy_type = strategy["type"]

            if strategy_type == "none":
                print(f"\n[{name}] Baseline - no training needed")
                continue

            print(f"\n[{name}] Training...")
            model_output_dir = self.output_dir / name

            try:
                if strategy_type in ("sft_qcm", "sft_benchmark", "sft_chosen_rej", "sft_qcm_combined"):
                    train_sft(self.config, strategy, str(model_output_dir))
                elif strategy_type == "dpo":
                    train_dpo(self.config, strategy, str(model_output_dir))
                elif strategy_type == "dpo_benchmark":
                    # DPO training on benchmark dataset
                    from trainers.trainer_dpo import DPOTrainerWrapper
                    trainer = DPOTrainerWrapper(self.config)
                    trainer.load_model()
                    benchmark_name = strategy.get("benchmark")
                    trainer.train_benchmark(
                        benchmark_name=benchmark_name,
                        output_dir=str(model_output_dir),
                        use_wandb=self.config.get("pipeline", {}).get("use_wandb", True),
                        max_samples=self.config.get("training", {}).get("train_samples"),
                        strategy_name=name
                    )
                elif strategy_type == "dpo_qcm":
                    # DPO training on QCM dataset (correct answer as chosen, random wrong as rejected)
                    from trainers.trainer_dpo import DPOTrainerWrapper
                    trainer = DPOTrainerWrapper(self.config)
                    trainer.load_model()
                    base_path = self.base_path
                    dataset_path = base_path / strategy["dataset"]
                    image_dir = base_path / strategy["image_dir"]
                    trainer.train_qcm(
                        dataset_path=str(dataset_path),
                        image_dir=str(image_dir),
                        output_dir=str(model_output_dir),
                        use_wandb=self.config.get("pipeline", {}).get("use_wandb", True),
                        max_samples=self.config.get("training", {}).get("train_samples"),
                        strategy_name=name
                    )
                elif strategy_type == "sft_qcm_chosen_rej":
                    # First train with QCM
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_sft(self.config, qcm_strategy, str(qcm_output))

                    # Then SFT on chosen/rejected dataset (using chosen responses)
                    chosen_rej_sft_strategy = {
                        "type": "sft_chosen_rej",
                        "dataset": strategy["chosen_rej_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, chosen_rej_sft_strategy, str(model_output_dir), base_model=str(qcm_output))
                elif strategy_type == "qcm_then_dpo":
                    # First train with QCM
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_sft(self.config, qcm_strategy, str(qcm_output))

                    # Then DPO on top
                    dpo_strategy = {
                        "type": "dpo",
                        "dataset": strategy["dpo_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_dpo(self.config, dpo_strategy, str(model_output_dir), base_model=str(qcm_output))
                elif strategy_type == "dpo_then_qcm":
                    # First train with DPO
                    dpo_strategy = {
                        "type": "dpo",
                        "dataset": strategy["dpo_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    dpo_output = model_output_dir / "dpo_stage"
                    train_dpo(self.config, dpo_strategy, str(dpo_output))

                    # Then QCM on top
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, qcm_strategy, str(model_output_dir), base_model=str(dpo_output))
                elif strategy_type == "sft_chosen_rej_qcm":
                    # Reverse order: SFT on chosen/rejected first, then QCM
                    chosen_rej_sft_strategy = {
                        "type": "sft_chosen_rej",
                        "dataset": strategy["chosen_rej_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    chosen_rej_output = model_output_dir / "chosen_rej_stage"
                    train_sft(self.config, chosen_rej_sft_strategy, str(chosen_rej_output))

                    # Then QCM on top
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, qcm_strategy, str(model_output_dir), base_model=str(chosen_rej_output))
                elif strategy_type == "sft_qcm_chosen_rej_combined":
                    # Combined: QCM (both) then SFT on chosen/rejected (both)
                    qcm_strategy = {
                        "type": "sft_qcm_combined",
                        "datasets": strategy["qcm_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_sft(self.config, qcm_strategy, str(qcm_output))

                    # Then SFT on chosen/rejected combined
                    chosen_rej_sft_strategy = {
                        "type": "sft_chosen_rej_combined",
                        "datasets": strategy["chosen_rej_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, chosen_rej_sft_strategy, str(model_output_dir), base_model=str(qcm_output))
                elif strategy_type == "sft_chosen_rej_qcm_combined":
                    # Combined reverse: SFT on chosen/rejected (both) then QCM (both)
                    chosen_rej_sft_strategy = {
                        "type": "sft_chosen_rej_combined",
                        "datasets": strategy["chosen_rej_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    chosen_rej_output = model_output_dir / "chosen_rej_stage"
                    train_sft(self.config, chosen_rej_sft_strategy, str(chosen_rej_output))

                    # Then QCM combined
                    qcm_strategy = {
                        "type": "sft_qcm_combined",
                        "datasets": strategy["qcm_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, qcm_strategy, str(model_output_dir), base_model=str(chosen_rej_output))
                # Full fine-tuning strategies (no LoRA, trains all parameters)
                elif strategy_type in ("full_ft_qcm", "full_ft_benchmark", "full_ft_chosen_rej_sft", "full_ft_qcm_combined", "full_ft_dpo"):
                    train_full_finetune(self.config, strategy, str(model_output_dir))
                elif strategy_type == "full_ft_qcm_then_dpo":
                    # Full fine-tune QCM first, then DPO
                    qcm_strategy = {
                        "type": "full_ft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_full_finetune(self.config, qcm_strategy, str(qcm_output))

                    # Then DPO on top
                    dpo_strategy = {
                        "type": "dpo",
                        "dataset": strategy["dpo_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_dpo(self.config, dpo_strategy, str(model_output_dir), base_model=str(qcm_output))
                else:
                    print(f"  Unknown strategy type: {strategy_type}")
                    continue

                print(f"  Model saved to: {model_output_dir}")

            except Exception as e:
                logger.error(f"Training failed for {name}: {e}")
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    raise
                print(f"  ERROR: {e}")

    def _run_evaluation_phase(self, strategies: List[Dict]):
        """Run evaluation for all models"""
        print("\n" + "=" * 80)
        print("PHASE 2: EVALUATION")
        print("=" * 80)

        from evaluators import EvaluatorAll

        evaluator = EvaluatorAll(self.config, str(self.cache_dir))

        for strategy in strategies:
            name = strategy["name"]
            strategy_type = strategy["type"]

            print(f"\n[{name}] Evaluating...")

            if strategy_type == "none":
                model_path = None
            else:
                model_path = self.output_dir / name

            try:
                results = evaluator.evaluate_all(
                    model_path=str(model_path) if model_path else None,
                    model_name=name
                )

                self.all_results[name] = results

                # Print summary
                if "summary" in results:
                    summary = results["summary"]
                    if "avg_benchmark_accuracy" in summary:
                        print(f"  Avg benchmark accuracy: {summary['avg_benchmark_accuracy']:.2f}%")
                    if "erp_qcm_accuracy" in summary:
                        print(f"  ERP QCM accuracy: {summary['erp_qcm_accuracy']:.2f}%")

                # Save individual results (mark as baseline if this is the baseline strategy)
                is_baseline = (strategy_type == "none")
                result_file = self.results_dir / f"{name}_{self.timestamp}.json"
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)

                # Also save to baseline_results.json if this is the baseline
                if is_baseline:
                    evaluator.save_results(results, str(self.results_dir), is_baseline=True)

            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    raise
                print(f"  ERROR: {e}")

    def _run_bertscore_phase(self, strategies: List[Dict]):
        """Run BERTScore evaluation for all models at the end"""
        # Check if BERTScore is enabled
        bertscore_config = self.config.get("evaluation", {}).get("erp_evaluation", {}).get("bertscore", {})
        if not bertscore_config.get("enabled", False):
            print("\n[BERTScore] Disabled in config, skipping")
            return

        print("\n" + "=" * 80)
        print("PHASE 5: BERTSCORE EVALUATION")
        print("=" * 80)
        print("Running BERTScore on all models (this may take a while)...")

        from evaluators import EvaluatorAll

        evaluator = EvaluatorAll(self.config, str(self.cache_dir))
        bertscore_results = {}

        for strategy in strategies:
            name = strategy["name"]
            strategy_type = strategy["type"]

            print(f"\n[{name}] Running BERTScore...")

            if strategy_type == "none":
                model_path = None
            else:
                model_path = self.output_dir / name
                # Skip if model doesn't exist
                if not model_path.exists():
                    print(f"  Skipping {name} - model not found at {model_path}")
                    continue

            try:
                result = evaluator.evaluate_bertscore_only(
                    model_path=str(model_path) if model_path else None,
                    model_name=name
                )

                bertscore_results[name] = result

                # Update the main results with BERTScore data
                if name in self.all_results:
                    if "erp_evaluation" not in self.all_results[name]:
                        self.all_results[name]["erp_evaluation"] = {}
                    if "bertscore" in result:
                        self.all_results[name]["erp_evaluation"]["bertscore"] = result["bertscore"]

                # Print summary
                if "bertscore" in result and "f1" in result["bertscore"]:
                    print(f"  BERTScore F1: {result['bertscore']['f1']:.4f}")

                # Update the saved result file
                result_file = self.results_dir / f"{name}_{self.timestamp}.json"
                if result_file.exists() and name in self.all_results:
                    with open(result_file, 'w') as f:
                        json.dump(self.all_results[name], f, indent=2)

            except Exception as e:
                logger.error(f"BERTScore failed for {name}: {e}")
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    raise
                print(f"  ERROR: {e}")

        # Save combined BERTScore results
        bertscore_file = self.results_dir / f"bertscore_all_{self.timestamp}.json"
        with open(bertscore_file, 'w') as f:
            json.dump(bertscore_results, f, indent=2)
        print(f"\nBERTScore results saved to: {bertscore_file}")

    def _save_sample_outputs(self, strategies: List[Dict], num_samples: int = 10):
        """Save random sample outputs from each model on each dataset"""
        print("\n" + "=" * 80)
        print("PHASE 3: SAMPLE OUTPUTS")
        print("=" * 80)
        print(f"Saving {num_samples} random samples per model/dataset...")

        # Create samples directory
        samples_dir = self.results_dir / "samples" / self.timestamp
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Get QCM datasets from config
        erp_eval = self.config.get("evaluation", {}).get("erp_evaluation", {})
        qcm_datasets = []

        for name, cfg in erp_eval.items():
            if name.startswith("qcm_") and isinstance(cfg, dict) and cfg.get("enabled", True):
                dataset_path = cfg.get("dataset")
                image_dir = cfg.get("image_dir", "datasets/images")
                if dataset_path:
                    qcm_datasets.append({
                        "name": name,
                        "dataset": dataset_path,
                        "image_dir": image_dir
                    })

        if not qcm_datasets:
            print("No QCM datasets found in config")
            return

        from evaluators.simple_evaluator import SimpleEvaluator
        from PIL import Image

        for strategy in strategies:
            model_name = strategy["name"]
            strategy_type = strategy["type"]

            print(f"\n[{model_name}] Generating sample outputs...")

            # Get model path - skip baseline (no trained model)
            if strategy_type == "none":
                print(f"  Skipping baseline - no trained model")
                continue
            else:
                model_path = self.output_dir / model_name
                if not model_path.exists():
                    print(f"  Skipping - model not found at {model_path}")
                    continue

            try:
                # Load model once for this strategy
                evaluator = SimpleEvaluator(str(self.cache_dir))
                evaluator.load_model(str(model_path) if model_path else None)

                for qcm_cfg in qcm_datasets:
                    dataset_name = qcm_cfg["name"]
                    dataset_path = self.base_path / qcm_cfg["dataset"]
                    image_dir = self.base_path / qcm_cfg["image_dir"]

                    if not dataset_path.exists():
                        print(f"  Skipping {dataset_name} - dataset not found")
                        continue

                    # Load dataset
                    with open(dataset_path, 'r') as f:
                        data = json.load(f)

                    # Sample random items
                    sample_size = min(num_samples, len(data))
                    random.seed(42)  # Fixed seed for reproducibility
                    samples = random.sample(data, sample_size)

                    sample_results = []
                    for i, item in enumerate(samples):
                        question = item.get("question", "")
                        options = item.get("options", {})
                        correct_answer = item.get("correct_answer", "")
                        image_path = item.get("image_path", "")

                        # Format question with options
                        options_text = "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
                        prompt = f"{question}\n\n{options_text}\n\nAnswer with the letter only."

                        # Load image if available
                        image = None
                        full_image_path = None
                        if image_path:
                            full_image_path = image_dir / image_path
                            if full_image_path.exists():
                                try:
                                    image = Image.open(full_image_path).convert("RGB")
                                except Exception:
                                    pass

                        # Get model response
                        try:
                            response = evaluator.generate(prompt, image)
                        except Exception as e:
                            response = f"[ERROR: {e}]"

                        sample_results.append({
                            "index": i + 1,
                            "question": question,
                            "options": options,
                            "correct_answer": correct_answer,
                            "model_response": response,
                            "image_path": str(image_path) if image_path else None
                        })

                    # Save samples for this model/dataset combination
                    output_file = samples_dir / f"{model_name}_{dataset_name}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "model": model_name,
                            "dataset": dataset_name,
                            "num_samples": len(sample_results),
                            "samples": sample_results
                        }, f, indent=2, ensure_ascii=False)

                    print(f"  Saved {len(sample_results)} samples for {dataset_name}")

                # Clean up model
                del evaluator

            except Exception as e:
                logger.error(f"Sample generation failed for {model_name}: {e}")
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    raise
                print(f"  ERROR: {e}")

        print(f"\nSample outputs saved to: {samples_dir}")

    def _load_existing_results(self):
        """Load existing evaluation results from result files"""
        print("\n" + "=" * 80)
        print("LOADING EXISTING RESULTS")
        print("=" * 80)

        # Get all JSON result files (exclude comparison files)
        result_files = [f for f in sorted(self.results_dir.glob("*_*.json"))
                       if not f.stem.startswith("comparison_")]

        if not result_files:
            logger.warning("No existing result files found in results directory")
            return

        # Group files by model name (files are named like "baseline_20251203_101937.json")
        model_files = {}
        for file_path in result_files:
            # Extract model name (everything before the last timestamp)
            filename = file_path.stem  # removes .json
            # Split by underscore and find where timestamp starts (YYYYMMDD_HHMMSS)
            parts = filename.split('_')

            # Find the timestamp parts (8 digits followed by 6 digits)
            timestamp_idx = None
            for i in range(len(parts) - 1):
                if len(parts[i]) == 8 and parts[i].isdigit() and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                    timestamp_idx = i
                    break

            if timestamp_idx is not None:
                model_name = '_'.join(parts[:timestamp_idx])
                timestamp_str = '_'.join(parts[timestamp_idx:])

                if model_name not in model_files or timestamp_str > model_files[model_name][1]:
                    model_files[model_name] = (file_path, timestamp_str)

        # Load the most recent result file for each model
        for model_name, (file_path, _) in model_files.items():
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    self.all_results[model_name] = results
                    print(f"  Loaded: {model_name} from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")

        print(f"\nLoaded {len(self.all_results)} model results")

    def _generate_comparison(self):
        """Generate comparison report"""
        print("\n" + "=" * 80)
        print("PHASE 4: COMPARISON")
        print("=" * 80)

        if not self.all_results:
            print("No results to compare")
            return

        # Build comparison table
        comparison_data = []

        for model_name, results in self.all_results.items():
            row = {"model": model_name}

            # Add benchmark accuracies
            benchmarks = results.get("benchmarks", {})
            for bench_name, bench_data in benchmarks.items():
                if "accuracy" in bench_data:
                    row[f"{bench_name}_acc"] = bench_data["accuracy"]

            # Add ERP-specific evaluations
            erp = results.get("erp_evaluation", {})

            # QCM Gemini
            if "qcm_gemini" in erp and "accuracy" in erp["qcm_gemini"]:
                row["qcm_gemini_acc"] = erp["qcm_gemini"]["accuracy"]

            # QCM Nova
            if "qcm_nova" in erp and "accuracy" in erp["qcm_nova"]:
                row["qcm_nova_acc"] = erp["qcm_nova"]["accuracy"]

            # QCM Claudette
            if "qcm_claudette" in erp and "accuracy" in erp["qcm_claudette"]:
                row["qcm_claudette_acc"] = erp["qcm_claudette"]["accuracy"]

            # QCM Procedure
            if "qcm_procedure1" in erp and "accuracy" in erp["qcm_procedure1"]:
                row["qcm_procedure1_acc"] = erp["qcm_procedure1"]["accuracy"]
            if "qcm_procedure2" in erp and "accuracy" in erp["qcm_procedure2"]:
                row["qcm_procedure2_acc"] = erp["qcm_procedure2"]["accuracy"]

            # DPO LogProb
            if "dpo_logprobs" in erp and "accuracy" in erp["dpo_logprobs"]:
                row["dpo_logprob_acc"] = erp["dpo_logprobs"]["accuracy"]

            # BERTScore (F1 as accuracy)
            if "bertscore" in erp and "f1" in erp["bertscore"]:
                row["bertscore_f1"] = erp["bertscore"]["f1"] * 100  # Convert to percentage

            # Calculate average
            accuracies = [v for k, v in row.items() if k.endswith("_acc")]
            if accuracies:
                row["avg_accuracy"] = sum(accuracies) / len(accuracies)

            comparison_data.append(row)

        if not comparison_data:
            print("No comparison data available")
            return

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Sort by avg_accuracy if available, otherwise just display as-is
        if "avg_accuracy" in df.columns:
            df = df.sort_values("avg_accuracy", ascending=False)
        elif len(df.columns) > 1:
            # Sort by model name if no accuracy columns
            df = df.sort_values("model")

        print("\nModel Comparison:")
        print("-" * 80)
        print(df.to_string(index=False))

        # Save comparison
        csv_path = self.results_dir / f"comparison_{self.timestamp}.csv"
        json_path = self.results_dir / f"comparison_{self.timestamp}.json"

        df.to_csv(csv_path, index=False)
        with open(json_path, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "comparison": df.to_dict(orient="records"),
                "all_results": self.all_results
            }, f, indent=2)

        print(f"\nComparison saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")

        # Generate insights
        self._generate_insights(df)

    def _generate_insights(self, df: pd.DataFrame):
        """Generate insights from comparison"""
        print("\n" + "-" * 80)
        print("INSIGHTS")
        print("-" * 80)

        if len(df) == 0:
            print("\nNo data available for insights")
            return

        if "avg_accuracy" not in df.columns:
            print("\nNo accuracy data available for insights")
            print(f"Models evaluated: {', '.join(df['model'].tolist())}")
            return

        # Best model
        best = df.iloc[0]
        print(f"\nBest model: {best['model']}")
        if pd.notna(best.get("avg_accuracy")):
            print(f"  Average accuracy: {best['avg_accuracy']:.2f}%")

        # Baseline comparison
        baseline = df[df["model"] == "baseline"]
        if len(baseline) > 0 and pd.notna(baseline.iloc[0].get("avg_accuracy")):
            baseline_avg = baseline.iloc[0]["avg_accuracy"]
            print(f"\nBaseline average: {baseline_avg:.2f}%")

            print("\nImprovement over baseline:")
            for _, row in df.iterrows():
                if row["model"] != "baseline" and pd.notna(row.get("avg_accuracy")):
                    improvement = row["avg_accuracy"] - baseline_avg
                    symbol = "+" if improvement > 0 else ""
                    print(f"  {row['model']}: {symbol}{improvement:.2f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NoHallucinations Pipeline")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    parser.add_argument("--eval-only", action="store_true",
                       help="Run evaluation only (skip training)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Generate comparison only (skip training and evaluation)")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (10 samples)")

    args = parser.parse_args()

    # Create pipeline
    pipeline = Pipeline(config_path=args.config)

    # Apply debug mode if requested
    if args.debug:
        pipeline.config["pipeline"]["debug_mode"] = True

    # Run pipeline
    pipeline.run(eval_only=args.eval_only, compare_only=args.compare_only)


if __name__ == "__main__":
    main()
