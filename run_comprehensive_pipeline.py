#!/usr/bin/env python3
"""
COMPREHENSIVE PIPELINE - The Ultimate Systematic Evaluation

This is the mega-pipeline that runs EVERYTHING:

1. Baseline: Test base model on all benchmarks
2. Benchmark Training (SFT): Train on each benchmark (DocVQA, OCRBench, etc.), test on all
3. ERP Training (SFT): Train on ERP with QCM, test on all benchmarks
4. ERP Training (DPO): Train on ERP with DPO, test on all benchmarks
5. ERP Training (SFT+DPO): Train on ERP with QCM then DPO, test on all
6. MEGA COMPARISON: Compare all models across all benchmarks

This answers:
- Which benchmarks benefit from training?
- Does ERP training hurt general VQA?
- Is SFT better than DPO for our use case?
- What's the best overall strategy?
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


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


BENCHMARKS_TO_TRAIN = ["docvqa", "ocrbench", "chartqa"]
# Use only 3 core benchmarks (document/visual understanding, irrelevant to ERP)
ALL_BENCHMARKS = ["ocrbench", "docvqa", "chartqa"]


class ComprehensivePipeline:
    """Orchestrates multiple systematic pipelines for ultimate comparison"""

    def __init__(self, args):
        self.args = args
        self.results_dir = Path("comprehensive_results")
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Track all experiment results
        self.all_experiment_results = {}

        # Setup logging
        self.log_file = self.results_dir / f"pipeline_log_{self.timestamp}.txt"
        self.tee = TeeOutput(self.log_file)
        sys.stdout = self.tee
        sys.stderr = self.tee

        print(f"="*80)
        print(f"Comprehensive Pipeline Started: {datetime.now()}")
        print(f"Log file: {self.log_file}")
        print(f"="*80)

    def run_systematic_pipeline(self, experiment_name: str, extra_args: list = None):
        """Run the systematic pipeline with specific configuration"""
        print("\n" + "🔬 "*20)
        print(f"EXPERIMENT: {experiment_name}")
        print("🔬 "*20)

        cmd = ["python3", "run_systematic_benchmark_pipeline.py"]

        # Add common args
        cmd.extend([
            "--benchmark-percentage", str(self.args.benchmark_percentage),
        ])

        if self.args.num_samples:
            cmd.extend(["--num-samples", str(self.args.num_samples)])

        if self.args.test_mode:
            cmd.append("--test-mode")

        if self.args.no_wandb:
            cmd.append("--no-wandb")

        if self.args.continue_on_error:
            cmd.append("--continue-on-error")

        # Add experiment-specific args
        if extra_args:
            cmd.extend(extra_args)

        print(f"\nCommand: {' '.join(cmd)}\n")

        if self.args.dry_run:
            print("DRY RUN - Command not executed")
            return True

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"❌ Experiment failed: {experiment_name}")
            if not self.args.continue_on_error:
                sys.exit(1)
            return False

        print(f"✅ Experiment completed: {experiment_name}")
        return True

    def phase1_baseline(self):
        """Phase 1: Baseline - test base model on everything"""
        print("\n" + "="*80)
        print("PHASE 1: BASELINE")
        print("Test base model on ALL benchmarks")
        print("="*80)

        if self.args.skip_baseline:
            print("Skipping baseline")
            return True

        return self.run_systematic_pipeline(
            experiment_name="baseline",
            extra_args=[]
        )

    def phase2_benchmark_training_sft(self):
        """Phase 2: Train on each benchmark (SFT), test on all"""
        print("\n" + "="*80)
        print("PHASE 2: BENCHMARK TRAINING (SFT)")
        print(f"Train on each of: {', '.join(BENCHMARKS_TO_TRAIN)}")
        print("Test each on ALL benchmarks")
        print("="*80)

        if self.args.skip_benchmark_training:
            print("Skipping benchmark training")
            return True

        benchmarks = self.args.train_benchmarks if self.args.train_benchmarks else BENCHMARKS_TO_TRAIN

        for benchmark in benchmarks:
            print(f"\n{'='*80}")
            print(f"Training on: {benchmark.upper()}")
            print(f"{'='*80}")

            success = self.run_systematic_pipeline(
                experiment_name=f"sft_on_{benchmark}",
                extra_args=[
                    "--skip-baseline",  # Already done in phase 1
                    "--train-benchmark", benchmark,
                    "--train-samples", str(self.args.train_samples),
                    "--epochs", str(self.args.epochs)
                ]
            )

            if not success and not self.args.continue_on_error:
                return False

        return True

    def phase3_erp_training_qcm(self):
        """Phase 3: Train on ERP with QCM (SFT), test on all"""
        print("\n" + "="*80)
        print("PHASE 3: ERP TRAINING - QCM (SFT)")
        print("Train on ERP QCM dataset, test on ALL benchmarks")
        print("="*80)

        if self.args.skip_erp_qcm:
            print("Skipping ERP QCM training")
            return True

        return self.run_systematic_pipeline(
            experiment_name="erp_qcm_sft",
            extra_args=[
                "--skip-baseline",
                "--train-erp",
                "--erp-strategy", "qcm",
                "--qcm-dataset", self.args.qcm_dataset,
                "--image-dir", self.args.image_dir,
                "--epochs", str(self.args.epochs)
            ]
        )

    def phase4_erp_training_dpo_dataset_sft(self):
        """Phase 4: Train on ERP DPO dataset with SFT (use chosen responses only), test on all"""
        print("\n" + "="*80)
        print("PHASE 4: ERP TRAINING - DPO Dataset with SFT")
        print("Train on ERP DPO dataset using SFT (chosen responses), test on ALL benchmarks")
        print("="*80)

        if self.args.skip_erp_dpo_sft:
            print("Skipping ERP DPO-dataset SFT training")
            return True

        return self.run_systematic_pipeline(
            experiment_name="erp_dpo_dataset_sft",
            extra_args=[
                "--skip-baseline",
                "--train-erp",
                "--erp-strategy", "dpo-sft",  # Use DPO dataset but train with SFT
                "--dpo-dataset", self.args.dpo_dataset,
                "--image-dir", self.args.image_dir,
                "--epochs", str(self.args.epochs)
            ]
        )

    def phase5_erp_training_dpo(self):
        """Phase 5: Train on ERP with DPO method, test on all"""
        print("\n" + "="*80)
        print("PHASE 5: ERP TRAINING - DPO Method")
        print("Train on ERP DPO dataset using DPO method, test on ALL benchmarks")
        print("="*80)

        if self.args.skip_erp_dpo:
            print("Skipping ERP DPO training")
            return True

        return self.run_systematic_pipeline(
            experiment_name="erp_dpo",
            extra_args=[
                "--skip-baseline",
                "--train-erp",
                "--erp-strategy", "dpo",
                "--dpo-dataset", self.args.dpo_dataset,
                "--image-dir", self.args.image_dir
            ]
        )

    def phase6_erp_training_combined(self):
        """Phase 6: Train on ERP with QCM+DPO, test on all"""
        print("\n" + "="*80)
        print("PHASE 6: ERP TRAINING - QCM + DPO (Sequential)")
        print("Train on ERP QCM then DPO, test on ALL benchmarks")
        print("="*80)

        if self.args.skip_erp_combined:
            print("Skipping ERP combined training")
            return True

        return self.run_systematic_pipeline(
            experiment_name="erp_qcm_dpo",
            extra_args=[
                "--skip-baseline",
                "--train-erp",
                "--erp-strategy", "qcm+dpo",
                "--qcm-dataset", self.args.qcm_dataset,
                "--dpo-dataset", self.args.dpo_dataset,
                "--image-dir", self.args.image_dir,
                "--epochs", str(self.args.epochs)
            ]
        )

    def phase7_mega_comparison(self):
        """Phase 6: Aggregate all results and create mega comparison"""
        print("\n" + "="*80)
        print("PHASE 6: MEGA COMPARISON")
        print("Aggregating and comparing ALL experiments")
        print("="*80)

        # Find all result files
        systematic_results_dir = Path("systematic_results")
        if not systematic_results_dir.exists():
            print("No systematic results found")
            return

        # Load all JSON results
        all_results = {}
        for json_file in systematic_results_dir.glob("systematic_comparison_*.json"):
            print(f"Loading: {json_file}")
            with open(json_file, 'r') as f:
                results = json.load(f)
                # Merge results
                all_results.update(results)

        if not all_results:
            print("No results to compare")
            return

        # Create mega comparison dataframe
        comparison_data = []

        for model_name, data in all_results.items():
            row = {"model": model_name}

            metrics = data.get("metrics", {})
            for benchmark, benchmark_metrics in metrics.items():
                row[f"{benchmark}_acc"] = benchmark_metrics.get("accuracy", 0)

            # Calculate average
            accuracies = [m.get("accuracy", 0) for m in metrics.values()]
            row["average_accuracy"] = sum(accuracies) / len(accuracies) if accuracies else 0.0

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        if len(df) == 0:
            print("No comparison data available")
            return

        # Sort by average accuracy
        df = df.sort_values("average_accuracy", ascending=False)

        # Print mega comparison
        print("\n" + "="*80)
        print("MEGA COMPARISON - ALL EXPERIMENTS")
        print("="*80 + "\n")

        # Show all models
        print(df.to_string(index=False))

        # Save comprehensive results
        csv_path = self.results_dir / f"mega_comparison_{self.timestamp}.csv"
        json_path = self.results_dir / f"mega_comparison_{self.timestamp}.json"

        df.to_csv(csv_path, index=False)

        mega_results = {
            "timestamp": self.timestamp,
            "all_models": all_results,
            "comparison": df.to_dict(orient="records")
        }

        with open(json_path, 'w') as f:
            json.dump(mega_results, f, indent=2)

        print(f"\n✅ Mega comparison saved:")
        print(f"   CSV:  {csv_path}")
        print(f"   JSON: {json_path}")

        # Generate insights
        self.generate_mega_insights(df)

    def generate_mega_insights(self, df: pd.DataFrame):
        """Generate comprehensive insights from all experiments"""
        print("\n" + "="*80)
        print("COMPREHENSIVE INSIGHTS")
        print("="*80)

        if len(df) == 0:
            return

        # 1. Best overall model
        best_model = df.iloc[0]["model"]
        best_acc = df.iloc[0]["average_accuracy"]
        print(f"\n🏆 BEST OVERALL MODEL: {best_model}")
        print(f"   Average Accuracy: {best_acc:.2f}%")

        # 2. Baseline comparison
        if "base_model" in df["model"].values:
            baseline_acc = df[df["model"] == "base_model"]["average_accuracy"].values[0]
            print(f"\n📊 IMPROVEMENT OVER BASELINE ({baseline_acc:.2f}%):")

            improvements = []
            for _, row in df.iterrows():
                if row["model"] != "base_model":
                    improvement = row["average_accuracy"] - baseline_acc
                    improvements.append({
                        "model": row["model"],
                        "improvement": improvement
                    })

            # Sort by improvement
            improvements.sort(key=lambda x: x["improvement"], reverse=True)

            for item in improvements[:10]:  # Top 10
                symbol = "📈" if item["improvement"] > 0 else "📉" if item["improvement"] < 0 else "➡️"
                print(f"   {symbol} {item['model']:40s}: {item['improvement']:+.2f}%")

        # 3. Compare training strategies
        print(f"\n🎯 TRAINING STRATEGY COMPARISON:")

        # Group models by strategy
        strategies = {
            "Benchmark SFT": [m for m in df["model"].values if "trained_on_" in m and "erp" not in m],
            "ERP QCM (SFT)": [m for m in df["model"].values if "erp" in m and "qcm" in m and "dpo" not in m],
            "ERP DPO-dataset (SFT)": [m for m in df["model"].values if "erp_dpo_dataset_sft" in m],
            "ERP DPO (Method)": [m for m in df["model"].values if "erp_dpo" in m and "dataset_sft" not in m and "qcm" not in m],
            "ERP Combined (QCM+DPO)": [m for m in df["model"].values if "erp" in m and "qcm_dpo" in m]
        }

        for strategy_name, models in strategies.items():
            if models:
                accs = [df[df["model"] == m]["average_accuracy"].values[0] for m in models]
                avg_acc = np.mean(accs)
                max_acc = np.max(accs)
                min_acc = np.min(accs)

                print(f"\n   {strategy_name}:")
                print(f"      Average: {avg_acc:.2f}%  |  Best: {max_acc:.2f}%  |  Worst: {min_acc:.2f}%")

                if len(models) > 1:
                    for model in models:
                        acc = df[df["model"] == model]["average_accuracy"].values[0]
                        print(f"      - {model}: {acc:.2f}%")

        # 4. Benchmark-specific insights
        print(f"\n📈 BEST MODEL PER BENCHMARK:")

        benchmark_cols = [col for col in df.columns if col.endswith("_acc")]
        for col in benchmark_cols:
            benchmark_name = col.replace("_acc", "")
            if col in df.columns:
                best_row = df.nlargest(1, col).iloc[0]
                best_model_for_bench = best_row["model"]
                best_acc_for_bench = best_row[col]

                # Check if trained on this benchmark
                is_trained_on_it = benchmark_name in best_model_for_bench

                indicator = "🎯" if is_trained_on_it else "✨"
                status = "SPECIALIZED" if is_trained_on_it else "TRANSFER"

                print(f"   {indicator} {benchmark_name:12s}: {best_model_for_bench:40s} ({best_acc_for_bench:.1f}%) [{status}]")

        # 5. SFT vs DPO comparison for ERP
        print(f"\n⚔️  SFT vs DPO FOR ERP:")

        erp_qcm_models = [m for m in df["model"].values if "erp" in m and "qcm" in m and "dpo" not in m]
        erp_dpo_sft_models = [m for m in df["model"].values if "erp_dpo_dataset_sft" in m]
        erp_dpo_models = [m for m in df["model"].values if "erp_dpo" in m and "dataset_sft" not in m and "qcm" not in m]
        erp_combined = [m for m in df["model"].values if "erp" in m and "qcm_dpo" in m]

        if erp_qcm_models:
            qcm_acc = df[df["model"].isin(erp_qcm_models)]["average_accuracy"].mean()
            print(f"   SFT on QCM dataset:        {qcm_acc:.2f}%")

        if erp_dpo_sft_models:
            dpo_sft_acc = df[df["model"].isin(erp_dpo_sft_models)]["average_accuracy"].mean()
            print(f"   SFT on DPO dataset:        {dpo_sft_acc:.2f}%")

        if erp_dpo_models:
            dpo_acc = df[df["model"].isin(erp_dpo_models)]["average_accuracy"].mean()
            print(f"   DPO method on DPO dataset: {dpo_acc:.2f}%")

        if erp_combined:
            combined_acc = df[df["model"].isin(erp_combined)]["average_accuracy"].mean()
            print(f"   Combined (QCM+DPO): {combined_acc:.2f}%")

        if erp_qcm_models and erp_dpo_models:
            qcm_acc = df[df["model"].isin(erp_qcm_models)]["average_accuracy"].mean()
            dpo_acc = df[df["model"].isin(erp_dpo_models)]["average_accuracy"].mean()

            if qcm_acc > dpo_acc:
                print(f"\n   🏆 Winner: SFT (QCM) is better by {qcm_acc - dpo_acc:.2f}%")
            elif dpo_acc > qcm_acc:
                print(f"\n   🏆 Winner: DPO is better by {dpo_acc - qcm_acc:.2f}%")
            else:
                print(f"\n   🤝 Tie: SFT and DPO perform equally")

        print("\n" + "="*80)

    def run_comprehensive_pipeline(self):
        """Run the complete comprehensive pipeline"""
        print("\n" + "🚀 "*20)
        print("COMPREHENSIVE PIPELINE - THE ULTIMATE EVALUATION")
        print("🚀 "*20)
        print("\nThis will run:")
        print("  1. Baseline on all benchmarks")
        print(f"  2. Train on {len(BENCHMARKS_TO_TRAIN)} benchmarks, test each on all")
        print("  3. Train on ERP (QCM with SFT), test on all")
        print("  4. Train on ERP (DPO dataset with SFT), test on all")
        print("  5. Train on ERP (DPO dataset with DPO method), test on all")
        print("  6. Train on ERP (QCM+DPO combined), test on all")
        print("  7. MEGA comparison of all results")
        print("\n" + "🚀 "*20)

        start_time = datetime.now()

        # Phase 1: Baseline
        self.phase1_baseline()

        # Phase 2: Benchmark training (SFT)
        self.phase2_benchmark_training_sft()

        # Phase 3: ERP QCM (SFT)
        self.phase3_erp_training_qcm()

        # Phase 4: ERP DPO dataset with SFT
        self.phase4_erp_training_dpo_dataset_sft()

        # Phase 5: ERP DPO (actual DPO training)
        self.phase5_erp_training_dpo()

        # Phase 6: ERP Combined
        self.phase6_erp_training_combined()

        # Phase 7: Mega comparison
        self.phase7_mega_comparison()

        # Summary
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE PIPELINE COMPLETED 🎉")
        print("="*80)
        print(f"Total time: {elapsed}")
        print(f"Results: {self.results_dir}")
        print(f"Full log: {self.log_file}")
        print("="*80 + "\n")

        # Restore stdout/stderr and close log
        sys.stdout = self.tee.terminal
        sys.stderr = self.tee.terminal
        self.tee.close()


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline - runs ALL experiments and comparisons"
    )

    # What to skip
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline benchmarking")
    parser.add_argument("--skip-benchmark-training", action="store_true",
                       help="Skip training on benchmarks")
    parser.add_argument("--skip-erp-qcm", action="store_true",
                       help="Skip ERP QCM training")
    parser.add_argument("--skip-erp-dpo-sft", action="store_true",
                       help="Skip ERP DPO dataset with SFT training")
    parser.add_argument("--skip-erp-dpo", action="store_true",
                       help="Skip ERP DPO method training")
    parser.add_argument("--skip-erp-combined", action="store_true",
                       help="Skip ERP combined training")

    # Training options
    parser.add_argument("--train-benchmarks", nargs="+",
                       choices=BENCHMARKS_TO_TRAIN,
                       help="Specific benchmarks to train on (default: all)")
    parser.add_argument("--train-samples", type=int, default=1000,
                       help="Max samples for benchmark training (default: 1000, use full dataset)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")

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
    parser.add_argument("--benchmark-percentage", type=float, default=100.0,
                       help="Percentage of benchmark data (automatically falls back on download errors)")
    parser.add_argument("--num-samples", type=int,
                       help="Number of samples per benchmark")

    # Pipeline options
    parser.add_argument("--test-mode", action="store_true",
                       help="Quick test mode")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode - use only 10 samples for everything")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands only")
    parser.add_argument("--continue-on-error", action="store_true",
                       help="Continue on errors")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable WandB")

    args = parser.parse_args()

    # Apply debug mode settings
    if args.debug:
        print("\n" + "🐛 "*20)
        print("DEBUG MODE ENABLED - Using 10 samples for everything")
        print("🐛 "*20 + "\n")
        args.train_samples = 10
        args.num_samples = 10
        args.epochs = 1
        args.benchmark_percentage = 1.0
        args.continue_on_error = True

    # Run comprehensive pipeline
    pipeline = ComprehensivePipeline(args)
    pipeline.run_comprehensive_pipeline()


if __name__ == "__main__":
    main()
