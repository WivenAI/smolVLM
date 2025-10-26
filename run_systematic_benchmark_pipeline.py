#!/usr/bin/env python3
"""
Systematic Benchmark Pipeline with WandB Integration

This pipeline runs systematic experiments:
1. Baseline: Test base model on ALL benchmarks
2. Benchmark Training: Train on ONE benchmark, test on ALL benchmarks
3. ERP Training: Train on ERP data, test on ALL benchmarks
4. Comparison: Compare all results with WandB visualizations

Use cases:
- Does training on DocVQA improve DocVQA? (sanity check)
- Does training on DocVQA improve other benchmarks? (transfer learning)
- Does ERP training maintain general VQA performance?
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import wandb


class TeeOutput:
    """Capture terminal output to both stdout and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """Check if the terminal is a TTY"""
        return hasattr(self.terminal, 'isatty') and self.terminal.isatty()

    def close(self):
        self.log.close()


# Use only 4 core benchmarks (document/visual understanding, relevant to ERP)
BENCHMARKS = ["ocrbench", "textvqa", "docvqa", "chartqa"]


class SystematicBenchmarkPipeline:
    """Systematic training and evaluation pipeline with WandB tracking"""

    def __init__(self, args):
        self.args = args
        self.results_dir = Path("systematic_results")
        self.results_dir.mkdir(exist_ok=True)

        self.all_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging
        self.log_file = self.results_dir / f"systematic_log_{self.timestamp}.txt"
        self.tee = TeeOutput(self.log_file)
        sys.stdout = self.tee
        sys.stderr = self.tee

        print(f"="*80)
        print(f"Systematic Pipeline Started: {datetime.now()}")
        print(f"Log file: {self.log_file}")
        print(f"="*80)

        # Initialize WandB for overall pipeline tracking
        if not args.no_wandb:
            wandb.init(
                project="SmallVLM-Systematic",
                name=f"systematic_pipeline_{self.timestamp}",
                config={
                    "benchmark_percentage": args.benchmark_percentage,
                    "num_samples": args.num_samples,
                    "epochs": args.epochs,
                    "train_benchmark": args.train_benchmark,
                    "train_erp": args.train_erp,
                }
            )

    def run_command(self, cmd: list, description: str):
        """Run a command and handle errors"""
        print("\n" + "="*80)
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print("="*80 + "\n")

        if self.args.dry_run:
            print("DRY RUN - Command not executed")
            return True

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"❌ Error running {description}")
            if not self.args.continue_on_error:
                sys.exit(1)
            return False

        print(f"✅ Completed: {description}")
        return True

    def benchmark_model(self, model_name: str, model_path: str, benchmarks: list = None):
        """Benchmark a model on specified benchmarks"""
        if benchmarks is None:
            benchmarks = self.args.benchmarks if self.args.benchmarks else BENCHMARKS

        print("\n" + "📊 "*20)
        print(f"Benchmarking: {model_name}")
        print(f"Model path: {model_path}")
        print(f"Benchmarks: {', '.join(benchmarks)}")
        print("📊 "*20)

        output_file = self.results_dir / f"{model_name}_{self.timestamp}.json"

        cmd = [
            "python3", "evaluate_ocrbench.py",
            "--model-path", model_path,
            "--output-file", str(output_file),
            "--percentage", str(self.args.benchmark_percentage)
        ]

        if benchmarks:
            cmd.extend(["--benchmarks"] + benchmarks)

        if self.args.num_samples:
            cmd.extend(["--num-samples", str(self.args.num_samples)])

        success = self.run_command(cmd, f"Benchmark {model_name}")

        # Load results
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                results = json.load(f)

            # Calculate metrics for each benchmark
            metrics = {}
            for benchmark_name, benchmark_results in results.items():
                if isinstance(benchmark_results, list) and benchmark_results:
                    accuracy = self.calculate_accuracy(benchmark_results)
                    metrics[benchmark_name] = {
                        "accuracy": accuracy,
                        "num_samples": len(benchmark_results)
                    }

            self.all_results[model_name] = {
                "metrics": metrics,
                "raw_results": results
            }

            # Log to WandB
            if not self.args.no_wandb:
                wandb_metrics = {f"{model_name}/{bench}": metrics[bench]["accuracy"]
                               for bench in metrics}
                wandb.log(wandb_metrics)

        return success

    def calculate_accuracy(self, results: list) -> float:
        """Calculate accuracy from benchmark results"""
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                gt = str(result['ground_truth']).lower().strip()
                response = str(result['response']).lower().strip()

                if gt in response or response in gt:
                    correct += 1
                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def phase1_baseline(self):
        """Phase 1: Benchmark base model on ALL benchmarks"""
        print("\n" + "🔍 "*20)
        print("PHASE 1: BASELINE - Test Base Model on ALL Benchmarks")
        print("This establishes baseline performance before any training")
        print("🔍 "*20)

        if self.args.skip_baseline:
            print("Skipping baseline (--skip-baseline)")
            return True

        return self.benchmark_model(
            model_name="base_model",
            model_path="HuggingFaceTB/SmolVLM-500M-Instruct",
            benchmarks=self.args.benchmarks if self.args.benchmarks else BENCHMARKS
        )

    def phase2_train_on_benchmark(self):
        """Phase 2: Train on ONE benchmark, test on ALL benchmarks"""
        if not self.args.train_benchmark or self.args.skip_benchmark_training:
            print("\nSkipping benchmark training phase")
            return True

        benchmark = self.args.train_benchmark

        print("\n" + "🎓 "*20)
        print(f"PHASE 2: BENCHMARK TRAINING - Train on {benchmark.upper()}")
        print(f"Then test on ALL benchmarks to measure:")
        print(f"  1. Does training improve {benchmark}? (should: YES)")
        print(f"  2. Does it improve related benchmarks? (transfer learning)")
        print(f"  3. Does it hurt unrelated benchmarks? (catastrophic forgetting)")
        print("🎓 "*20)

        # Step 1: Train on benchmark
        model_output = self.results_dir / f"trained_on_{benchmark}"

        cmd = [
            "python3", "finetune_on_benchmarks.py",
            "--benchmark", benchmark,
            "--output-dir", str(model_output),
            "--max-samples", str(self.args.train_samples),
            "--num-epochs", str(self.args.epochs)
        ]

        if self.args.test_mode:
            cmd.append("--test")

        if not self.run_command(cmd, f"Train on {benchmark}"):
            return False

        # Step 2: Benchmark trained model on ALL benchmarks
        return self.benchmark_model(
            model_name=f"trained_on_{benchmark}",
            model_path=str(model_output),
            benchmarks=self.args.benchmarks if self.args.benchmarks else BENCHMARKS
        )

    def phase3_train_on_erp(self):
        """Phase 3: Train on ERP data, test on ALL benchmarks"""
        if not self.args.train_erp or self.args.skip_erp_training:
            print("\nSkipping ERP training phase")
            return True

        print("\n" + "🏢 "*20)
        print("PHASE 3: ERP TRAINING - Train on ERP Data")
        print("Then test on ALL benchmarks to measure:")
        print("  1. Does ERP training improve ERP tasks?")
        print("  2. Does it maintain general VQA performance?")
        print("  3. Is there catastrophic forgetting?")
        print("🏢 "*20)

        training_strategy = self.args.erp_strategy

        # Train based on strategy
        if training_strategy == "qcm":
            model_output = self.results_dir / "trained_on_erp_qcm"
            cmd = [
                "python3", "finetune_smolvlm_qcm.py",
                "--output-dir", str(model_output),
                "--dataset", self.args.qcm_dataset,
                "--image-dir", self.args.image_dir,
                "--num-epochs", str(self.args.epochs)
            ]
        elif training_strategy == "dpo-sft":
            # Use DPO dataset but train with SFT (use only chosen responses)
            model_output = self.results_dir / "trained_on_erp_dpo_dataset_sft"
            cmd = [
                "python3", "finetune_smolvlm_qcm.py",  # Use QCM script which does SFT
                "--output-dir", str(model_output),
                "--dataset", self.args.dpo_dataset,  # But use DPO dataset
                "--image-dir", self.args.image_dir,
                "--num-epochs", str(self.args.epochs),
                "--use-dpo-chosen-only"  # Flag to use only chosen responses from DPO dataset
            ]
        elif training_strategy == "dpo":
            model_output = self.results_dir / "trained_on_erp_dpo"
            cmd = [
                "python3", "finetune_smolvlm_dpo.py",
                "--output-dir", str(model_output),
                "--dataset", self.args.dpo_dataset,
                "--image-dir", self.args.image_dir
            ]
        elif training_strategy == "qcm+dpo":
            # First QCM, then DPO
            qcm_output = self.results_dir / "trained_on_erp_qcm_temp"

            cmd_qcm = [
                "python3", "finetune_smolvlm_qcm.py",
                "--output-dir", str(qcm_output),
                "--dataset", self.args.qcm_dataset,
                "--image-dir", self.args.image_dir,
                "--num-epochs", str(self.args.epochs)
            ]

            if not self.run_command(cmd_qcm, "ERP QCM Training"):
                return False

            model_output = self.results_dir / "trained_on_erp_qcm_dpo"
            cmd = [
                "python3", "finetune_smolvlm_dpo.py",
                "--base-model", str(qcm_output),
                "--output-dir", str(model_output),
                "--dataset", self.args.dpo_dataset,
                "--image-dir", self.args.image_dir
            ]
        else:
            print(f"Unknown ERP strategy: {training_strategy}")
            return False

        if self.args.test_mode:
            cmd.append("--test")

        if not self.run_command(cmd, f"ERP Training ({training_strategy})"):
            return False

        # Benchmark on ALL benchmarks
        return self.benchmark_model(
            model_name=f"trained_on_erp_{training_strategy}",
            model_path=str(model_output),
            benchmarks=self.args.benchmarks if self.args.benchmarks else BENCHMARKS
        )

    def phase4_comparison(self):
        """Phase 4: Generate comprehensive comparison report"""
        print("\n" + "📈 "*20)
        print("PHASE 4: COMPARISON & ANALYSIS")
        print("📈 "*20)

        if not self.all_results:
            print("No results to compare")
            return

        # Create comparison dataframe
        comparison_data = []

        for model_name, data in self.all_results.items():
            row = {"model": model_name}

            metrics = data.get("metrics", {})
            for benchmark, benchmark_metrics in metrics.items():
                row[f"{benchmark}_acc"] = benchmark_metrics["accuracy"]
                row[f"{benchmark}_samples"] = benchmark_metrics["num_samples"]

            # Calculate average
            accuracies = [m["accuracy"] for m in metrics.values()]
            row["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by average accuracy
        df = df.sort_values("average_accuracy", ascending=False)

        # Print comparison table
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON RESULTS")
        print("="*80 + "\n")

        # Print summary table
        summary_cols = ["model", "average_accuracy"] + [f"{b}_acc" for b in BENCHMARKS if f"{b}_acc" in df.columns]
        print(df[summary_cols].to_string(index=False))

        # Save results
        csv_path = self.results_dir / f"systematic_comparison_{self.timestamp}.csv"
        json_path = self.results_dir / f"systematic_comparison_{self.timestamp}.json"

        df.to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"\n✅ Results saved:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")

        # Analysis insights
        self.print_insights(df)

        # Log final comparison to WandB
        if not self.args.no_wandb:
            # Create WandB table
            table = wandb.Table(dataframe=df)
            wandb.log({"comparison_table": table})

            # Log summary metrics
            for _, row in df.iterrows():
                model_name = row["model"]
                wandb.run.summary[f"{model_name}_avg"] = row["average_accuracy"]

    def print_insights(self, df: pd.DataFrame):
        """Print key insights from comparison"""
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)

        if len(df) == 0:
            print("No data to analyze")
            return

        # Best model
        best_model = df.iloc[0]["model"]
        best_accuracy = df.iloc[0]["average_accuracy"]
        print(f"\n🏆 Best Overall Model: {best_model}")
        print(f"   Average Accuracy: {best_accuracy:.2f}%")

        # Compare to baseline
        if "base_model" in df["model"].values:
            baseline_acc = df[df["model"] == "base_model"]["average_accuracy"].values[0]

            print(f"\n📊 Comparison to Baseline ({baseline_acc:.2f}%):")
            for _, row in df.iterrows():
                if row["model"] != "base_model":
                    improvement = row["average_accuracy"] - baseline_acc
                    symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    print(f"   {symbol} {row['model']:30s}: {improvement:+.2f}%")

        # Benchmark-specific insights
        if self.args.train_benchmark:
            benchmark = self.args.train_benchmark
            trained_model = f"trained_on_{benchmark}"

            if trained_model in df["model"].values and "base_model" in df["model"].values:
                print(f"\n🎯 Impact of Training on {benchmark.upper()}:")

                base_row = df[df["model"] == "base_model"].iloc[0]
                trained_row = df[df["model"] == trained_model].iloc[0]

                for bench in BENCHMARKS:
                    col = f"{bench}_acc"
                    if col in df.columns:
                        base_acc = base_row.get(col, 0)
                        trained_acc = trained_row.get(col, 0)
                        improvement = trained_acc - base_acc

                        if bench == benchmark:
                            status = "✅ TARGET" if improvement > 5 else "⚠️ UNEXPECTED"
                        elif improvement > 2:
                            status = "✨ TRANSFER"
                        elif improvement < -2:
                            status = "⚠️ FORGOT"
                        else:
                            status = "➡️ STABLE"

                        print(f"   {status:15s} {bench:12s}: {base_acc:5.1f}% → {trained_acc:5.1f}% ({improvement:+.1f}%)")

        print("\n" + "="*80)

    def run_full_pipeline(self):
        """Run complete systematic pipeline"""
        print("\n" + "🔬 "*20)
        print("SYSTEMATIC BENCHMARK PIPELINE")
        print("🔬 "*20)

        start_time = datetime.now()

        # Phase 1: Baseline
        self.phase1_baseline()

        # Phase 2: Benchmark training
        if self.args.train_benchmark:
            self.phase2_train_on_benchmark()

        # Phase 3: ERP training
        if self.args.train_erp:
            self.phase3_train_on_erp()

        # Phase 4: Comparison
        self.phase4_comparison()

        # Summary
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"Total time: {elapsed}")
        print(f"Results: {self.results_dir}")
        print(f"Full log: {self.log_file}")

        if not self.args.no_wandb:
            print(f"WandB: {wandb.run.url}")

        print("="*80 + "\n")

        # Restore stdout/stderr and close log
        sys.stdout = self.tee.terminal
        sys.stderr = self.tee.terminal
        self.tee.close()

        # Finish WandB
        if not self.args.no_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Systematic benchmark training and evaluation pipeline"
    )

    # What to run
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline benchmarking")
    parser.add_argument("--train-benchmark", type=str,
                       choices=["docvqa", "ocrbench", "textvqa", "chartqa"],
                       help="Train on this benchmark, then test on all")
    parser.add_argument("--skip-benchmark-training", action="store_true",
                       help="Skip benchmark training phase")
    parser.add_argument("--train-erp", action="store_true",
                       help="Train on ERP data, then test on all benchmarks")
    parser.add_argument("--skip-erp-training", action="store_true",
                       help="Skip ERP training phase")

    # Training options
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--train-samples", type=int, default=500,
                       help="Max samples for benchmark training")
    parser.add_argument("--erp-strategy", type=str,
                       choices=["qcm", "dpo-sft", "dpo", "qcm+dpo"],
                       default="qcm+dpo",
                       help="ERP training strategy: qcm (SFT on QCM), dpo-sft (SFT on DPO dataset), dpo (DPO method), qcm+dpo (both)")

    # ERP dataset paths
    parser.add_argument("--qcm-dataset", type=str,
                       default="balanced_qcm_all_end.json",
                       help="QCM dataset path")
    parser.add_argument("--dpo-dataset", type=str,
                       default="dpo_image_dataset/dpo_dataset.json",
                       help="DPO dataset path")
    parser.add_argument("--image-dir", type=str,
                       default="dpo_image_dataset",
                       help="Image directory")

    # Evaluation options
    parser.add_argument("--benchmarks", nargs="+",
                       choices=BENCHMARKS,
                       help="Specific benchmarks to evaluate (default: all)")
    parser.add_argument("--benchmark-percentage", type=float, default=10.0,
                       help="Percentage of benchmark data to use")
    parser.add_argument("--num-samples", type=int,
                       help="Number of samples per benchmark (overrides percentage)")

    # Pipeline options
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with minimal samples")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue even if a step fails")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB logging")

    args = parser.parse_args()

    # Validate
    if args.benchmark_percentage <= 0 or args.benchmark_percentage > 100:
        parser.error("--benchmark-percentage must be between 1 and 100")

    # Run pipeline
    pipeline = SystematicBenchmarkPipeline(args)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
