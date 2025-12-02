#!/usr/bin/env python3
"""
NoHallucinations Pipeline - Clean, modular training and evaluation pipeline

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

        logger.info(f"Loaded config from: {self.config_path}")
        return config

    def _apply_debug_mode(self):
        """Apply debug mode settings"""
        if self.config.get("pipeline", {}).get("debug_mode", False):
            logger.info("DEBUG MODE - Using 10 samples for everything")
            self.config["training"]["train_samples"] = 10
            self.config["training"]["epochs"] = 1
            for bench in self.config.get("evaluation", {}).get("benchmarks", []):
                bench["max_samples"] = 10

    def run(self, eval_only: bool = False) -> Dict[str, Any]:
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

        print(f"\nEnabled training strategies: {len(enabled_strategies)}")
        for s in enabled_strategies:
            print(f"  - {s['name']} ({s['type']})")

        # Phase 1: Training (if not eval-only)
        if not eval_only:
            self._run_training_phase(enabled_strategies)

        # Phase 2: Evaluation
        self._run_evaluation_phase(enabled_strategies)

        # Phase 3: Comparison
        self._generate_comparison()

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

        from train.trainer_sft import train_sft
        from train.trainer_dpo import train_dpo

        for strategy in strategies:
            name = strategy["name"]
            strategy_type = strategy["type"]

            if strategy_type == "none":
                print(f"\n[{name}] Baseline - no training needed")
                continue

            print(f"\n[{name}] Training...")
            model_output_dir = self.output_dir / name

            try:
                if strategy_type in ("sft_qcm", "sft_benchmark", "sft_dpo", "sft_qcm_combined"):
                    train_sft(self.config, strategy, str(model_output_dir))
                elif strategy_type == "dpo":
                    train_dpo(self.config, strategy, str(model_output_dir))
                elif strategy_type == "sft_qcm_dpo":
                    # First train with QCM
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_sft(self.config, qcm_strategy, str(qcm_output))

                    # Then SFT on DPO dataset (using chosen responses)
                    dpo_sft_strategy = {
                        "type": "sft_dpo",
                        "dataset": strategy["dpo_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, dpo_sft_strategy, str(model_output_dir), base_model=str(qcm_output))
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
                elif strategy_type == "sft_dpo_qcm":
                    # Reverse order: SFT-DPO first, then QCM
                    dpo_sft_strategy = {
                        "type": "sft_dpo",
                        "dataset": strategy["dpo_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    dpo_output = model_output_dir / "dpo_stage"
                    train_sft(self.config, dpo_sft_strategy, str(dpo_output))

                    # Then QCM on top
                    qcm_strategy = {
                        "type": "sft_qcm",
                        "dataset": strategy["qcm_dataset"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, qcm_strategy, str(model_output_dir), base_model=str(dpo_output))
                elif strategy_type == "sft_qcm_dpo_combined":
                    # Combined: QCM (both) then SFT-DPO (both)
                    qcm_strategy = {
                        "type": "sft_qcm_combined",
                        "datasets": strategy["qcm_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    qcm_output = model_output_dir / "qcm_stage"
                    train_sft(self.config, qcm_strategy, str(qcm_output))

                    # Then SFT-DPO combined
                    dpo_sft_strategy = {
                        "type": "sft_dpo_combined",
                        "datasets": strategy["dpo_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, dpo_sft_strategy, str(model_output_dir), base_model=str(qcm_output))
                elif strategy_type == "sft_dpo_qcm_combined":
                    # Combined reverse: SFT-DPO (both) then QCM (both)
                    dpo_sft_strategy = {
                        "type": "sft_dpo_combined",
                        "datasets": strategy["dpo_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    dpo_output = model_output_dir / "dpo_stage"
                    train_sft(self.config, dpo_sft_strategy, str(dpo_output))

                    # Then QCM combined
                    qcm_strategy = {
                        "type": "sft_qcm_combined",
                        "datasets": strategy["qcm_datasets"],
                        "image_dir": strategy["image_dir"]
                    }
                    train_sft(self.config, qcm_strategy, str(model_output_dir), base_model=str(dpo_output))
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

        from eval import EvaluatorAll

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

                # Save individual results
                result_file = self.results_dir / f"{name}_{self.timestamp}.json"
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                logger.error(f"Evaluation failed for {name}: {e}")
                if not self.config.get("pipeline", {}).get("continue_on_error", True):
                    raise
                print(f"  ERROR: {e}")

    def _generate_comparison(self):
        """Generate comparison report"""
        print("\n" + "=" * 80)
        print("PHASE 3: COMPARISON")
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

            # Add ERP QCM accuracy
            erp = results.get("erp_evaluation", {})
            if "qcm" in erp and "accuracy" in erp["qcm"]:
                row["erp_qcm_acc"] = erp["qcm"]["accuracy"]

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
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode (10 samples)")

    args = parser.parse_args()

    # Create pipeline
    pipeline = Pipeline(config_path=args.config)

    # Apply debug mode if requested
    if args.debug:
        pipeline.config["pipeline"]["debug_mode"] = True

    # Run pipeline
    pipeline.run(eval_only=args.eval_only)


if __name__ == "__main__":
    main()
