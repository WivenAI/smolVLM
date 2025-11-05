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


# Use only 3 core benchmarks (document/visual understanding, relevant to ERP)
# Note: textvqa removed because it falls back to docvqa for training (would be redundant)
BENCHMARKS = ["ocrbench", "docvqa", "chartqa"]


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

    def evaluate_erp_qcm(self, model_name: str, model_path: str):
        """Evaluate model on ERP QCM dataset with comprehensive metrics"""
        print("\n" + "🏢 "*20)
        print(f"ERP QCM Evaluation: {model_name}")
        print(f"Model path: {model_path}")
        print("🏢 "*20)

        output_file = self.results_dir / f"{model_name}_erp_qcm_{self.timestamp}.json"

        # Build command for ERP QCM evaluation
        cmd = [
            "python3", "evaluate_erp_qcm.py",
            "--model-path", model_path,
            "--output-file", str(output_file)
        ]

        # Add dataset paths if specified
        if hasattr(self.args, 'qcm_dataset') and self.args.qcm_dataset:
            cmd.extend(["--dataset", self.args.qcm_dataset])

        if hasattr(self.args, 'image_dir') and self.args.image_dir:
            cmd.extend(["--image-dir", self.args.image_dir])

        if self.args.num_samples:
            cmd.extend(["--max-samples", str(self.args.num_samples)])

        success = self.run_command(cmd, f"ERP QCM Evaluation {model_name}")

        # Load results
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                erp_results = json.load(f)

            # Extract metrics
            erp_metrics = {
                "accuracy": erp_results['metrics']['accuracy'],
                "avg_log_prob": erp_results['metrics'].get('avg_log_prob', None),
                "bertscore_f1": erp_results['metrics']['bertscore']['f1'] if erp_results['metrics'].get('bertscore') else None,
                "bertscore_precision": erp_results['metrics']['bertscore']['precision'] if erp_results['metrics'].get('bertscore') else None,
                "bertscore_recall": erp_results['metrics']['bertscore']['recall'] if erp_results['metrics'].get('bertscore') else None,
                "num_samples": erp_results['num_samples']
            }

            # Add to results
            if model_name in self.all_results:
                self.all_results[model_name]['metrics']['erp_qcm'] = erp_metrics
            else:
                self.all_results[model_name] = {
                    "metrics": {"erp_qcm": erp_metrics},
                    "raw_results": {}
                }

            # Log to WandB
            if not self.args.no_wandb:
                wandb_metrics = {
                    f"{model_name}/erp_qcm_accuracy": erp_metrics["accuracy"],
                }
                if erp_metrics["avg_log_prob"] is not None:
                    wandb_metrics[f"{model_name}/erp_qcm_log_prob"] = erp_metrics["avg_log_prob"]
                if erp_metrics["bertscore_f1"] is not None:
                    wandb_metrics[f"{model_name}/erp_qcm_bertscore_f1"] = erp_metrics["bertscore_f1"]

                wandb.log(wandb_metrics)

        return success

    def evaluate_erp_dpo(self, model_name: str, model_path: str):
        """Evaluate model on ERP DPO dataset with BERTScore and log-probability"""
        print("\n" + "🎯 "*20)
        print(f"ERP DPO Evaluation: {model_name}")
        print(f"Model path: {model_path}")
        print("🎯 "*20)

        output_file = self.results_dir / f"{model_name}_erp_dpo_{self.timestamp}.json"

        # Build command for ERP DPO evaluation
        cmd = [
            "python3", "evaluate_erp_dpo.py",
            "--model-path", model_path,
            "--output-file", str(output_file)
        ]

        # Add dataset paths if specified
        if hasattr(self.args, 'dpo_dataset') and self.args.dpo_dataset:
            cmd.extend(["--dataset", self.args.dpo_dataset])

        if hasattr(self.args, 'image_dir') and self.args.image_dir:
            cmd.extend(["--image-dir", self.args.image_dir])

        if self.args.num_samples:
            cmd.extend(["--max-samples", str(self.args.num_samples)])

        success = self.run_command(cmd, f"ERP DPO Evaluation {model_name}")

        # Load results
        if success and output_file.exists():
            with open(output_file, 'r') as f:
                dpo_results = json.load(f)

            # Extract metrics
            dpo_metrics = {
                "preference_accuracy": dpo_results['metrics']['preference_accuracy'],
                "avg_chosen_logprob": dpo_results['metrics']['avg_chosen_logprob'],
                "avg_rejected_logprob": dpo_results['metrics']['avg_rejected_logprob'],
                "avg_margin": dpo_results['metrics']['avg_margin'],
                "bertscore_f1": dpo_results['metrics']['bertscore']['f1'] if dpo_results['metrics'].get('bertscore') else None,
                "bertscore_precision": dpo_results['metrics']['bertscore']['precision'] if dpo_results['metrics'].get('bertscore') else None,
                "bertscore_recall": dpo_results['metrics']['bertscore']['recall'] if dpo_results['metrics'].get('bertscore') else None,
                "num_samples": dpo_results['num_samples']
            }

            # Add to results
            if model_name in self.all_results:
                self.all_results[model_name]['metrics']['erp_dpo'] = dpo_metrics
            else:
                self.all_results[model_name] = {
                    "metrics": {"erp_dpo": dpo_metrics},
                    "raw_results": {}
                }

            # Log to WandB
            if not self.args.no_wandb:
                wandb_metrics = {
                    f"{model_name}/erp_dpo_preference_acc": dpo_metrics["preference_accuracy"],
                    f"{model_name}/erp_dpo_margin": dpo_metrics["avg_margin"],
                }
                if dpo_metrics["bertscore_f1"] is not None:
                    wandb_metrics[f"{model_name}/erp_dpo_bertscore_f1"] = dpo_metrics["bertscore_f1"]

                wandb.log(wandb_metrics)

        return success

    def benchmark_model(self, model_name: str, model_path: str, benchmarks: list = None):
        """Benchmark a model on specified benchmarks and ERP QCM"""
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
                    accuracy = self.calculate_accuracy(benchmark_results, benchmark_name=benchmark_name)
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

        # ALSO evaluate on ERP datasets
        if not self.args.skip_erp_eval:
            # Evaluate on QCM (accuracy)
            self.evaluate_erp_qcm(model_name, model_path)
            # Evaluate on DPO (BERTScore and log-probability)
            self.evaluate_erp_dpo(model_name, model_path)

        return success

    def calculate_accuracy(self, results: list, benchmark_name: str = None) -> float:
        """Calculate accuracy from benchmark results using benchmark-specific methods"""
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                # Check if any ground truth matches (benchmark-specific logic)
                is_correct = False
                for gt in ground_truths:
                    gt_str = str(gt).lower().strip()

                    # For OCRBench: check if ground truth is in response
                    if benchmark_name == 'ocrbench':
                        if gt_str in response:
                            is_correct = True
                            break
                    # For DocVQA/ChartQA/others: check both directions
                    else:
                        if gt_str in response or response in gt_str:
                            is_correct = True
                            break

                if is_correct:
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

            # Handle standard benchmarks
            benchmark_accuracies = []
            for benchmark, benchmark_metrics in metrics.items():
                if benchmark == "erp_qcm":
                    # Handle ERP QCM separately
                    row["erp_qcm_acc"] = benchmark_metrics["accuracy"]
                    row["erp_qcm_log_prob"] = benchmark_metrics.get("avg_log_prob", None)
                    row["erp_qcm_bertscore_f1"] = benchmark_metrics.get("bertscore_f1", None)
                    row["erp_qcm_samples"] = benchmark_metrics["num_samples"]
                elif benchmark == "erp_dpo":
                    # Handle ERP DPO separately
                    row["erp_dpo_pref_acc"] = benchmark_metrics["preference_accuracy"]
                    row["erp_dpo_margin"] = benchmark_metrics.get("avg_margin", None)
                    row["erp_dpo_bertscore_f1"] = benchmark_metrics.get("bertscore_f1", None)
                    row["erp_dpo_samples"] = benchmark_metrics["num_samples"]
                else:
                    # Standard benchmark
                    row[f"{benchmark}_acc"] = benchmark_metrics["accuracy"]
                    row[f"{benchmark}_samples"] = benchmark_metrics["num_samples"]
                    benchmark_accuracies.append(benchmark_metrics["accuracy"])

            # Calculate average (standard benchmarks only, not including ERP QCM)
            row["average_accuracy"] = sum(benchmark_accuracies) / len(benchmark_accuracies) if benchmark_accuracies else 0.0

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

        # Add ERP QCM columns if present
        if "erp_qcm_acc" in df.columns:
            summary_cols.extend(["erp_qcm_acc", "erp_qcm_log_prob", "erp_qcm_bertscore_f1"])

        # Add ERP DPO columns if present
        if "erp_dpo_pref_acc" in df.columns:
            summary_cols.extend(["erp_dpo_pref_acc", "erp_dpo_margin", "erp_dpo_bertscore_f1"])

        # Only include columns that exist
        summary_cols = [col for col in summary_cols if col in df.columns]

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

        # ERP QCM insights
        if "erp_qcm_acc" in df.columns and "base_model" in df["model"].values:
            print(f"\n🏢 ERP QCM Performance:")

            base_row = df[df["model"] == "base_model"].iloc[0]
            base_erp_acc = base_row.get("erp_qcm_acc", None)
            base_erp_logprob = base_row.get("erp_qcm_log_prob", None)
            base_erp_bert = base_row.get("erp_qcm_bertscore_f1", None)

            if base_erp_acc is not None:
                print(f"   Baseline:")
                print(f"      Accuracy:       {base_erp_acc:.2f}%")
                if base_erp_logprob is not None:
                    print(f"      Avg Log Prob:   {base_erp_logprob:.4f}")
                if base_erp_bert is not None:
                    print(f"      BERTScore F1:   {base_erp_bert:.4f}")

                # Compare trained models
                print(f"\n   Trained Models:")
                for _, row in df.iterrows():
                    if row["model"] != "base_model" and "erp" in row["model"]:
                        model_erp_acc = row.get("erp_qcm_acc", None)
                        model_erp_logprob = row.get("erp_qcm_log_prob", None)
                        model_erp_bert = row.get("erp_qcm_bertscore_f1", None)

                        if model_erp_acc is not None:
                            acc_improvement = model_erp_acc - base_erp_acc
                            symbol = "📈" if acc_improvement > 0 else "📉" if acc_improvement < 0 else "➡️"

                            print(f"\n   {symbol} {row['model']}:")
                            print(f"      Accuracy:       {model_erp_acc:.2f}% ({acc_improvement:+.2f}%)")

                            if model_erp_logprob is not None and base_erp_logprob is not None:
                                logprob_improvement = model_erp_logprob - base_erp_logprob
                                print(f"      Avg Log Prob:   {model_erp_logprob:.4f} ({logprob_improvement:+.4f})")

                            if model_erp_bert is not None and base_erp_bert is not None:
                                bert_improvement = model_erp_bert - base_erp_bert
                                print(f"      BERTScore F1:   {model_erp_bert:.4f} ({bert_improvement:+.4f})")

        # ERP DPO insights
        if "erp_dpo_pref_acc" in df.columns and "base_model" in df["model"].values:
            print(f"\n🎯 ERP DPO Performance:")

            base_row = df[df["model"] == "base_model"].iloc[0]
            base_dpo_pref = base_row.get("erp_dpo_pref_acc", None)
            base_dpo_margin = base_row.get("erp_dpo_margin", None)
            base_dpo_bert = base_row.get("erp_dpo_bertscore_f1", None)

            if base_dpo_pref is not None:
                print(f"   Baseline:")
                print(f"      Preference Accuracy: {base_dpo_pref:.2f}%")
                if base_dpo_margin is not None:
                    print(f"      Margin:              {base_dpo_margin:.4f}")
                if base_dpo_bert is not None:
                    print(f"      BERTScore F1:        {base_dpo_bert:.4f}")

                # Compare trained models
                print(f"\n   Trained Models:")
                for _, row in df.iterrows():
                    if row["model"] != "base_model" and "erp" in row["model"]:
                        model_dpo_pref = row.get("erp_dpo_pref_acc", None)
                        model_dpo_margin = row.get("erp_dpo_margin", None)
                        model_dpo_bert = row.get("erp_dpo_bertscore_f1", None)

                        if model_dpo_pref is not None:
                            pref_improvement = model_dpo_pref - base_dpo_pref
                            symbol = "📈" if pref_improvement > 0 else "📉" if pref_improvement < 0 else "➡️"

                            print(f"\n   {symbol} {row['model']}:")
                            print(f"      Preference Accuracy: {model_dpo_pref:.2f}% ({pref_improvement:+.2f}%)")

                            if model_dpo_margin is not None and base_dpo_margin is not None:
                                margin_improvement = model_dpo_margin - base_dpo_margin
                                print(f"      Margin:              {model_dpo_margin:.4f} ({margin_improvement:+.4f})")

                            if model_dpo_bert is not None and base_dpo_bert is not None:
                                bert_improvement = model_dpo_bert - base_dpo_bert
                                print(f"      BERTScore F1:        {model_dpo_bert:.4f} ({bert_improvement:+.4f})")

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
                       choices=["docvqa", "ocrbench", "chartqa"],
                       help="Train on this benchmark, then test on all")
    parser.add_argument("--skip-benchmark-training", action="store_true",
                       help="Skip benchmark training phase")
    parser.add_argument("--train-erp", action="store_true",
                       help="Train on ERP data, then test on all benchmarks")
    parser.add_argument("--skip-erp-training", action="store_true",
                       help="Skip ERP training phase")
    parser.add_argument("--skip-erp-eval", action="store_true",
                       help="Skip ERP QCM evaluation for all models")

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
                       default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="DPO dataset path")
    parser.add_argument("--image-dir", type=str,
                       default="dpo_image_dataset",
                       help="Image directory")

    # Evaluation options
    parser.add_argument("--benchmarks", nargs="+",
                       choices=BENCHMARKS,
                       help="Specific benchmarks to evaluate (default: all)")
    parser.add_argument("--benchmark-percentage", type=float, default=100.0,
                       help="Percentage of benchmark data to use (automatically falls back on download errors)")
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
