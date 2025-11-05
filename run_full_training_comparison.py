#!/usr/bin/env python3
"""
Comprehensive Training and Benchmark Comparison Pipeline
Runs multiple training strategies and compares their performance:
1. Base model (no training) - baseline
2. QCM SFT only - supervised fine-tuning on QCM dataset
3. DPO only - direct preference optimization on DPO dataset
4. QCM SFT + DPO - sequential training (current approach)

After training, benchmarks all models and generates comparison report.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


class TrainingComparisonPipeline:
    """Orchestrates training and benchmarking of multiple model variants"""

    def __init__(self, args):
        self.args = args
        self.results_dir = Path("comparison_results")
        self.results_dir.mkdir(exist_ok=True)

        # Model output directories
        self.models = {
            "base": "HuggingFaceTB/SmolVLM-500M-Instruct",
            "benchmark_canary": "./smolvlm-docvqa-finetuned",  # Trained on DocVQA as canary
            "qcm_sft": "./smolvlm-500m-qcm-sft-only",
            "dpo_only": "./smolvlm-500m-dpo-only",
            "qcm_dpo_sequential": "./smolvlm-500m-qcm-dpo-sequential"
        }

        self.benchmark_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    def train_benchmark_canary(self):
        """Train on benchmark dataset (DocVQA) as canary to verify training works"""
        if self.args.skip_training or self.args.skip_benchmark_canary:
            print("Skipping benchmark canary training")
            return True

        print("\n" + "🐤 "*20)
        print("Training Strategy 0: Benchmark Canary (DocVQA)")
        print("This verifies training works on well-known datasets")
        print("🐤 "*20)

        cmd = [
            "python3", "finetune_on_benchmarks.py",
            "--benchmark", "docvqa",
            "--output-dir", self.models["benchmark_canary"],
            "--max-samples", "500",
            "--num-epochs", str(self.args.epochs)
        ]

        if self.args.test:
            cmd.append("--test")

        return self.run_command(cmd, "Benchmark Canary Training (DocVQA)")

    def train_qcm_sft_only(self):
        """Train model on QCM dataset using SFT only"""
        if self.args.skip_training:
            print("Skipping QCM SFT training (--skip-training)")
            return True

        print("\n" + "🎯 "*20)
        print("Training Strategy 1: QCM SFT Only")
        print("🎯 "*20)

        cmd = [
            "python3", "finetune_smolvlm_qcm.py",
            "--output-dir", self.models["qcm_sft"],
            "--dataset", "balanced_qcm_all_end.json",
            "--image-dir", "dpo_image_dataset",
            "--num-epochs", str(self.args.epochs)
        ]

        if self.args.test:
            cmd.append("--test")

        return self.run_command(cmd, "QCM SFT Training")

    def train_dpo_only(self):
        """Train model using DPO only (no QCM pre-training)"""
        if self.args.skip_training:
            print("Skipping DPO-only training (--skip-training)")
            return True

        print("\n" + "🎯 "*20)
        print("Training Strategy 2: DPO Only")
        print("🎯 "*20)

        cmd = [
            "python3", "finetune_smolvlm_dpo.py",
            "--output-dir", self.models["dpo_only"],
            "--dataset", "dpo_image_dataset/dpo_dataset_gemini.json",
            "--image-dir", "dpo_image_dataset"
        ]

        if self.args.test:
            cmd.append("--test")

        return self.run_command(cmd, "DPO-only Training")

    def train_qcm_then_dpo(self):
        """Train sequentially: QCM SFT first, then DPO"""
        if self.args.skip_training:
            print("Skipping sequential QCM+DPO training (--skip-training)")
            return True

        print("\n" + "🎯 "*20)
        print("Training Strategy 3: QCM SFT → DPO Sequential")
        print("🎯 "*20)

        # Step 1: Train QCM SFT
        print("\nStep 1/2: QCM SFT Training...")
        qcm_temp_dir = "./temp_qcm_for_sequential"
        cmd_qcm = [
            "python3", "finetune_smolvlm_qcm.py",
            "--output-dir", qcm_temp_dir,
            "--dataset", "balanced_qcm_all_end.json",
            "--image-dir", "dpo_image_dataset",
            "--num-epochs", str(self.args.epochs)
        ]

        if self.args.test:
            cmd_qcm.append("--test")

        if not self.run_command(cmd_qcm, "QCM SFT (Sequential Step 1/2)"):
            return False

        # Step 2: Continue with DPO from QCM checkpoint
        print("\nStep 2/2: DPO Training from QCM checkpoint...")
        cmd_dpo = [
            "python3", "finetune_smolvlm_dpo.py",
            "--base-model", qcm_temp_dir,
            "--output-dir", self.models["qcm_dpo_sequential"],
            "--dataset", "dpo_image_dataset/dpo_dataset_gemini.json",
            "--image-dir", "dpo_image_dataset"
        ]

        if self.args.test:
            cmd_dpo.append("--test")

        return self.run_command(cmd_dpo, "DPO Training (Sequential Step 2/2)")

    def benchmark_model(self, model_name: str, model_path: str):
        """Run benchmarks on a trained model"""
        print("\n" + "📊 "*20)
        print(f"Benchmarking: {model_name}")
        print(f"Model path: {model_path}")
        print("📊 "*20)

        output_file = self.results_dir / f"{model_name}_benchmark_{self.timestamp}.json"

        cmd = [
            "python3", "evaluate_ocrbench.py",
            "--model-path", model_path,
            "--output-file", str(output_file),
            "--percentage", str(self.args.benchmark_percentage)
        ]

        if self.args.benchmark_subset:
            cmd.extend(["--benchmarks"] + self.args.benchmark_subset)

        if self.args.num_samples:
            cmd.extend(["--num-samples", str(self.args.num_samples)])

        success = self.run_command(cmd, f"Benchmark {model_name}")

        if success and output_file.exists():
            with open(output_file, 'r') as f:
                self.benchmark_results[model_name] = json.load(f)

        return success

    def benchmark_all_models(self):
        """Benchmark all trained models"""
        print("\n" + "="*80)
        print("BENCHMARKING ALL MODELS")
        print("="*80)

        for model_name, model_path in self.models.items():
            # Skip base model if requested
            if model_name == "base" and self.args.skip_base_benchmark:
                print(f"Skipping base model benchmark (--skip-base-benchmark)")
                continue

            # Check if model exists (except base model)
            if model_name != "base" and not Path(model_path).exists():
                print(f"⚠️  Model not found: {model_path}, skipping benchmark")
                continue

            self.benchmark_model(model_name, model_path)

    def calculate_metrics(self, results: list) -> dict:
        """Calculate accuracy metrics from benchmark results"""
        if not results:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                gt = str(result['ground_truth']).lower().strip()
                response = str(result['response']).lower().strip()

                if gt in response or response in gt:
                    correct += 1
                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0
        return {"accuracy": accuracy, "total": total, "correct": correct}

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80)

        if not self.benchmark_results:
            print("No benchmark results available for comparison")
            return

        # Prepare comparison data
        comparison_data = []

        for model_name, results in self.benchmark_results.items():
            model_row = {"model": model_name}

            # Calculate accuracy for each benchmark
            for benchmark_name, benchmark_results in results.items():
                if isinstance(benchmark_results, list):
                    metrics = self.calculate_metrics(benchmark_results)
                    model_row[f"{benchmark_name}_accuracy"] = metrics["accuracy"]
                    model_row[f"{benchmark_name}_total"] = metrics["total"]

            # Calculate average across all benchmarks
            accuracies = [v for k, v in model_row.items() if k.endswith("_accuracy")]
            model_row["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0

            comparison_data.append(model_row)

        # Create DataFrame for nice formatting
        df = pd.DataFrame(comparison_data)

        # Sort by average accuracy
        df = df.sort_values("average_accuracy", ascending=False)

        # Print comparison table
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON RESULTS")
        print("="*80 + "\n")
        print(df.to_string(index=False))
        print("\n" + "="*80)

        # Save comparison to CSV and JSON
        csv_path = self.results_dir / f"comparison_summary_{self.timestamp}.csv"
        json_path = self.results_dir / f"comparison_summary_{self.timestamp}.json"

        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)

        print(f"\n✅ Comparison report saved:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")

        # Print key insights
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)

        best_model = df.iloc[0]["model"]
        best_accuracy = df.iloc[0]["average_accuracy"]

        print(f"\n🏆 Best performing model: {best_model}")
        print(f"   Average accuracy: {best_accuracy:.2f}%")

        if "base" in df["model"].values:
            base_accuracy = df[df["model"] == "base"]["average_accuracy"].values[0]
            improvement = best_accuracy - base_accuracy
            print(f"\n📈 Improvement over base model: {improvement:+.2f}%")

        # Compare training strategies
        print("\n📊 Training Strategy Comparison:")
        for _, row in df.iterrows():
            model_name = row["model"]
            avg_acc = row["average_accuracy"]

            strategy_desc = {
                "base": "No training (baseline)",
                "benchmark_canary": "Benchmark canary (DocVQA)",
                "qcm_sft": "QCM SFT only",
                "dpo_only": "DPO only",
                "qcm_dpo_sequential": "QCM SFT → DPO sequential"
            }.get(model_name, model_name)

            print(f"   {strategy_desc:30s}: {avg_acc:6.2f}%")

        print("\n" + "="*80)

    def run_canary_test(self):
        """Step 0: Run canary test to verify training works"""
        print("\n" + "🐤 "*20)
        print("STEP 0: CANARY TEST (Verify Training Works)")
        print("🐤 "*20)

        cmd = [
            "python3", "run_canary_test.py",
            "--num-samples", "3",
            "--epochs", "10",
            "--threshold", "70.0"
        ]

        success = self.run_command(cmd, "Canary Test")

        if not success:
            print("\n" + "="*80)
            print("⚠️  CANARY TEST FAILED!")
            print("⚠️  Training code has bugs. Fix before proceeding.")
            print("="*80)

            if not self.args.continue_on_error:
                print("\nUse --continue-on-error to proceed anyway (not recommended)")
                sys.exit(1)

        return success

    def run_full_pipeline(self):
        """Run the complete training and comparison pipeline"""
        print("\n" + "🚀 "*20)
        print("STARTING COMPREHENSIVE TRAINING COMPARISON PIPELINE")
        print("🚀 "*20)

        start_time = datetime.now()

        # Canary test phase (optional but recommended)
        if not self.args.skip_canary and not self.args.benchmark_only:
            print("\n" + "="*80)
            print("PHASE 0: CANARY TEST")
            print("="*80)
            self.run_canary_test()

        # Training phase
        if not self.args.benchmark_only:
            print("\n" + "="*80)
            print("PHASE 1: TRAINING")
            print("="*80)

            # Train all model variants
            if not self.args.skip_benchmark_canary:
                self.train_benchmark_canary()

            self.train_qcm_sft_only()
            self.train_dpo_only()
            self.train_qcm_then_dpo()

        # Benchmarking phase
        print("\n" + "="*80)
        print("PHASE 2: BENCHMARKING")
        print("="*80)

        self.benchmark_all_models()

        # Comparison phase
        print("\n" + "="*80)
        print("PHASE 3: COMPARISON")
        print("="*80)

        self.generate_comparison_report()

        # Summary
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"Total time: {elapsed}")
        print(f"Results saved in: {self.results_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive training and benchmarking comparison pipeline"
    )

    # Training options
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with limited samples")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training, only run benchmarks")
    parser.add_argument("--benchmark-only", action="store_true",
                       help="Only run benchmarks (alias for --skip-training)")
    parser.add_argument("--skip-canary", action="store_true",
                       help="Skip memorization canary test (not recommended)")
    parser.add_argument("--skip-benchmark-canary", action="store_true",
                       help="Skip benchmark canary training (DocVQA)")

    # Benchmarking options
    parser.add_argument("--skip-base-benchmark", action="store_true",
                       help="Skip benchmarking the base model")
    parser.add_argument("--benchmark-percentage", type=float, default=10.0,
                       help="Percentage of benchmark dataset to use (default: 10)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples per benchmark (overrides percentage)")
    parser.add_argument("--benchmark-subset", nargs="+",
                       choices=["ocrbench", "docvqa", "chartqa",
                               "ai2d", "scienceqa", "mmstar", "mmmu", "mathvista"],
                       help="Run only specific benchmarks (textvqa removed - redundant with docvqa)")

    # Pipeline options
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without executing")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue pipeline even if a step fails")

    args = parser.parse_args()

    # Validate arguments
    if args.benchmark_percentage <= 0 or args.benchmark_percentage > 100:
        parser.error("--benchmark-percentage must be between 1 and 100")

    # Run pipeline
    pipeline = TrainingComparisonPipeline(args)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
