#!/usr/bin/env python3
"""
Benchmark Pipeline for SmolVLM
1. Benchmark base model on 1000+ questions
2. Train model on DPO/SFT data
3. Benchmark trained model
4. Compare results
"""

import os
import json
import subprocess
import datetime
from pathlib import Path


class BenchmarkPipeline:
    def __init__(self):
        self.base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"
        self.results_dir = Path("./benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

        # Timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_benchmark(self, model_path: str, output_file: str, num_samples: int = 100):
        """Run benchmarks on working datasets"""
        print(f"\n{'='*80}")
        print(f"BENCHMARKING: {model_path}")
        print(f"{'='*80}\n")

        results = {}

        # 1. OCRBench
        print("\n[1/4] Running OCRBench...")
        cmd = [
            "python", "evaluate_ocrbench.py",
            "--model-path", model_path,
            "--benchmarks", "ocrbench",
            "--percentage", "10",  # 10% to get ~100 samples
            "--num-samples", str(num_samples),
            "--output-file", f"temp_ocr_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # 2. DocVQA
        print("\n[2/4] Running DocVQA...")
        cmd = [
            "python", "evaluate_ocrbench.py",
            "--model-path", model_path,
            "--benchmarks", "docvqa",
            "--percentage", "10",
            "--num-samples", str(num_samples),
            "--output-file", f"temp_doc_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # 3. ChartQA
        print("\n[3/4] Running ChartQA...")
        cmd = [
            "python", "evaluate_ocrbench.py",
            "--model-path", model_path,
            "--benchmarks", "chartqa",
            "--percentage", "10",
            "--num-samples", str(num_samples),
            "--output-file", f"temp_chart_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # 4. ERP QCM
        print("\n[4/5] Running ERP QCM...")
        cmd = [
            "python", "evaluate_qcm_erp.py",
            "--model-path", model_path,
            "--percentage", "10",
            "--num-samples", str(num_samples),
            "--output-file", f"temp_qcm_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # 5. DPO Log Probabilities
        print("\n[5/6] Running DPO Log Probability Benchmark...")
        cmd = [
            "python", "evaluate_dpo_logprobs.py",
            "--model-path", model_path,
            "--output-file", f"temp_dpo_logprob_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # 6. BERTScore
        print("\n[6/6] Running BERTScore Benchmark...")
        cmd = [
            "python", "evaluate_bertscore_dpo.py",
            "--model-path", model_path,
            "--output-file", f"temp_bertscore_{self.timestamp}.json"
        ]
        subprocess.run(cmd)

        # Combine results
        print("\nCombining results...")
        results = self._combine_results()

        # Save combined results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")
        return results

    def _combine_results(self):
        """Combine individual benchmark results"""
        combined = {
            "timestamp": self.timestamp,
            "benchmarks": {}
        }

        # Read temp files
        temp_files = [
            f"temp_ocr_{self.timestamp}.json",
            f"temp_doc_{self.timestamp}.json",
            f"temp_chart_{self.timestamp}.json",
            f"temp_qcm_{self.timestamp}.json",
            f"temp_dpo_logprob_{self.timestamp}.json",
            f"temp_bertscore_{self.timestamp}.json"
        ]

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file) as f:
                    data = json.load(f)
                    # Handle different result formats
                    if "results" in data:
                        # QCM format
                        combined["benchmarks"]["erp_qcm"] = {
                            "questions": data["results"],
                            "accuracy": data.get("accuracy", 0),
                            "total": data.get("total_questions", 0),
                            "correct": data.get("correct_answers", 0)
                        }
                    elif "overall_metrics" in data and "preference_accuracy" in data.get("overall_metrics", {}):
                        # DPO Log Probability format
                        combined["benchmarks"]["dpo_logprob"] = {
                            "overall_metrics": data["overall_metrics"],
                            "metadata": data.get("metadata", {}),
                            "preference_accuracy": data["overall_metrics"]["preference_accuracy"],
                            "margin_mean": data["overall_metrics"]["margin"]["mean"],
                            "num_examples": data["overall_metrics"]["num_examples"]
                        }
                    elif "overall_metrics" in data and "f1" in data.get("overall_metrics", {}):
                        # BERTScore format
                        combined["benchmarks"]["bertscore"] = {
                            "overall_metrics": data["overall_metrics"],
                            "metadata": data.get("metadata", {}),
                            "f1_mean": data["overall_metrics"]["f1"]["mean"],
                            "precision_mean": data["overall_metrics"]["precision"]["mean"],
                            "recall_mean": data["overall_metrics"]["recall"]["mean"],
                            "num_examples": data["metadata"]["num_examples"]
                        }
                    else:
                        # Vision benchmark format
                        combined["benchmarks"].update(data)
                os.remove(temp_file)  # Cleanup

        return combined

    def train_model(self, train_script: str = "finetune_smolvlm_lora.py",
                   output_dir: str = "./smolvlm-500m-lora-finetuned"):
        """Train the model using specified training script"""
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL")
        print(f"{'='*80}\n")

        print(f"Training script: {train_script}")
        print(f"Output directory: {output_dir}")

        # Check if training data exists
        if not os.path.exists("dpo_image_dataset/dpo_dataset.json"):
            print("\n⚠ Warning: Training dataset not found at dpo_image_dataset/dpo_dataset.json")
            print("Skipping training step...")
            return None

        # Run training
        cmd = ["python", train_script]
        print(f"\nRunning: {' '.join(cmd)}\n")

        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ Training completed. Model saved to: {output_dir}")
            return output_dir
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Training failed: {e}")
            return None

    def calculate_accuracy(self, results: dict) -> dict:
        """Calculate accuracy for each benchmark (manual string matching)"""
        accuracies = {}

        for benchmark_name, benchmark_data in results.get("benchmarks", {}).items():
            if not benchmark_data:
                continue

            # Handle ERP QCM format (already has accuracy calculated)
            if benchmark_name == "erp_qcm" and isinstance(benchmark_data, dict):
                if "accuracy" in benchmark_data:
                    accuracies[benchmark_name] = {
                        "correct": benchmark_data.get("correct", 0),
                        "total": benchmark_data.get("total", 0),
                        "accuracy": benchmark_data.get("accuracy", 0.0)
                    }
                    continue

            # Handle DPO Log Probability format
            if benchmark_name == "dpo_logprob" and isinstance(benchmark_data, dict):
                if "preference_accuracy" in benchmark_data:
                    num_examples = benchmark_data.get("num_examples", 0)
                    pref_acc = benchmark_data.get("preference_accuracy", 0.0)
                    correct = int(pref_acc * num_examples)
                    accuracies[benchmark_name] = {
                        "correct": correct,
                        "total": num_examples,
                        "accuracy": pref_acc * 100  # Convert to percentage
                    }
                    continue

            # Handle BERTScore format
            if benchmark_name == "bertscore" and isinstance(benchmark_data, dict):
                if "f1_mean" in benchmark_data:
                    num_examples = benchmark_data.get("num_examples", 0)
                    f1_score = benchmark_data.get("f1_mean", 0.0)
                    # Use F1 score as "accuracy" metric (it's already 0-1, convert to percentage)
                    accuracies[benchmark_name] = {
                        "correct": int(f1_score * num_examples),  # Approximate
                        "total": num_examples,
                        "accuracy": f1_score * 100  # Convert to percentage
                    }
                    continue

            # Handle vision benchmark format (list of questions)
            questions = benchmark_data if isinstance(benchmark_data, list) else []
            correct = 0
            total = len(questions)

            for q in questions:
                response = q.get('response', '').lower()
                ground_truth = q.get('ground_truth', '')

                # Handle different ground truth formats
                if isinstance(ground_truth, dict):
                    gt_text = ground_truth.get('text', '').lower()
                elif isinstance(ground_truth, list):
                    gt_text = [str(gt).lower() for gt in ground_truth]
                else:
                    gt_text = str(ground_truth).lower()

                # Check if answer is in response
                if isinstance(gt_text, list):
                    if any(gt in response for gt in gt_text):
                        correct += 1
                else:
                    if gt_text in response:
                        correct += 1

            accuracy = (correct / total * 100) if total > 0 else 0.0
            accuracies[benchmark_name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }

        return accuracies

    def compare_results(self, base_results: dict, trained_results: dict):
        """Compare base model vs trained model results"""
        print(f"\n{'='*80}")
        print("COMPARISON: Base Model vs Trained Model")
        print(f"{'='*80}\n")

        # Calculate accuracies
        base_acc = self.calculate_accuracy(base_results)
        trained_acc = self.calculate_accuracy(trained_results)

        # Print comparison table
        print(f"{'Benchmark':<15} {'Base Model':<15} {'Trained Model':<15} {'Improvement':<15}")
        print("-" * 70)

        improvements = {}
        for benchmark in base_acc.keys():
            base_val = base_acc[benchmark]['accuracy']
            trained_val = trained_acc.get(benchmark, {}).get('accuracy', 0.0)
            improvement = trained_val - base_val
            improvements[benchmark] = improvement

            improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
            print(f"{benchmark:<15} {base_val:>6.1f}%{'':<8} {trained_val:>6.1f}%{'':<8} {improvement_str:>10}")

        # Overall average
        avg_base = sum(b['accuracy'] for b in base_acc.values()) / len(base_acc) if base_acc else 0
        avg_trained = sum(t['accuracy'] for t in trained_acc.values()) / len(trained_acc) if trained_acc else 0
        avg_improvement = avg_trained - avg_base

        print("-" * 70)
        print(f"{'AVERAGE':<15} {avg_base:>6.1f}%{'':<8} {avg_trained:>6.1f}%{'':<8} {avg_improvement:>+6.1f}%")
        print("=" * 70)

        # Save comparison
        comparison = {
            "timestamp": self.timestamp,
            "base_model": self.base_model,
            "base_accuracies": base_acc,
            "trained_accuracies": trained_acc,
            "improvements": improvements,
            "average_improvement": avg_improvement
        }

        comparison_file = self.results_dir / f"comparison_{self.timestamp}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n✓ Comparison saved to: {comparison_file}\n")

        return comparison

    def run_full_pipeline(self, num_samples: int = 100, train: bool = True, skip_base: bool = False):
        """Run the complete pipeline"""
        print(f"\n{'#'*80}")
        print(f"# SmolVLM Benchmark Pipeline")
        print(f"# Timestamp: {self.timestamp}")
        print(f"# Samples per benchmark: {num_samples}")
        print(f"# Benchmarks: OCRBench, DocVQA, ChartQA, ERP QCM, DPO LogProb, BERTScore")
        print(f"# Total questions: ~{num_samples * 4} + DPO dataset (LogProb + BERTScore)")
        print(f"{'#'*80}\n")

        base_results = None
        base_acc = None

        # Step 1: Benchmark base model (optional)
        if not skip_base:
            base_results_file = self.results_dir / f"base_model_{self.timestamp}.json"
            print("\n" + "="*80)
            print("STEP 1: Benchmarking Base Model")
            print("="*80)
            base_results = self.run_benchmark(
                model_path=self.base_model,
                output_file=str(base_results_file),
                num_samples=num_samples
            )

            # Display base model results
            base_acc = self.calculate_accuracy(base_results)
            print(f"\nBase Model Results:")
            for benchmark, acc in base_acc.items():
                print(f"  {benchmark}: {acc['correct']}/{acc['total']} = {acc['accuracy']:.1f}%")
        else:
            print("\n⚠ Skipping base model benchmark (--train-only mode)")

        # Step 2: Train model (optional)
        if train:
            print("\n" + "="*80)
            print("STEP 2: Training Model")
            print("="*80)
            trained_model_path = self.train_model()

            if trained_model_path is None:
                print("\n⚠ Training skipped or failed. Pipeline stopped.")
                return base_results, None, None

            # Step 3: Benchmark trained model
            print("\n" + "="*80)
            print("STEP 3: Benchmarking Trained Model")
            print("="*80)
            trained_results_file = self.results_dir / f"trained_model_{self.timestamp}.json"
            trained_results = self.run_benchmark(
                model_path=trained_model_path,
                output_file=str(trained_results_file),
                num_samples=num_samples
            )

            # Step 4: Compare results (only if base model was benchmarked)
            if base_results:
                print("\n" + "="*80)
                print("STEP 4: Comparing Results")
                print("="*80)
                comparison = self.compare_results(base_results, trained_results)
                return base_results, trained_results, comparison
            else:
                print("\n⚠ Skipping comparison (no base model results)")
                # Display trained model results
                trained_acc = self.calculate_accuracy(trained_results)
                print(f"\nTrained Model Results:")
                for benchmark, acc in trained_acc.items():
                    print(f"  {benchmark}: {acc['correct']}/{acc['total']} = {acc['accuracy']:.1f}%")
                return None, trained_results, None
        else:
            print("\n⚠ Training skipped (train=False)")
            return base_results, None, None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SmolVLM Benchmark Pipeline")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Full pipeline command (default behavior)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full training and benchmarking pipeline')
    pipeline_parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples per benchmark (default: 100)")
    pipeline_parser.add_argument("--skip-training", action="store_true",
                       help="Skip training step, only benchmark base model")
    pipeline_parser.add_argument("--train-only", action="store_true",
                       help="Train and benchmark only the trained model (skip base model)")
    pipeline_parser.add_argument("--train-script", default="finetune_smolvlm_lora.py",
                       help="Training script to use (default: finetune_smolvlm_lora.py - recommended for GPUs <16GB)")

    # Compare command - benchmark both models and compare
    compare_parser = subparsers.add_parser('compare', help='Benchmark and compare base model vs finetuned model')
    compare_parser.add_argument("--num-samples", type=int, default=500,
                       help="Number of samples per benchmark (default: 500)")
    compare_parser.add_argument("--finetuned-model", type=str, default="./smolvlm-500m-lora-finetuned",
                       help="Path to finetuned model (default: ./smolvlm-500m-lora-finetuned)")
    compare_parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct",
                       help="Path to base model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")

    # Benchmark-only command - test a single model
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark a single model')
    benchmark_parser.add_argument("--model-path", type=str, required=True,
                       help="Path to model to benchmark")
    benchmark_parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples per benchmark (default: 100)")
    benchmark_parser.add_argument("--output-file", type=str, default=None,
                       help="Output file for results (default: auto-generated)")

    args = parser.parse_args()

    pipeline = BenchmarkPipeline()

    # Handle different commands
    if args.command == 'compare':
        # Compare base model vs finetuned model
        print(f"\n{'#'*80}")
        print(f"# SmolVLM Model Comparison")
        print(f"# Base Model: {args.base_model}")
        print(f"# Finetuned Model: {args.finetuned_model}")
        print(f"# Samples per benchmark: {args.num_samples}")
        print(f"{'#'*80}\n")

        # Benchmark base model
        base_results_file = pipeline.results_dir / f"base_model_{pipeline.timestamp}.json"
        print("\n" + "="*80)
        print("STEP 1/3: Benchmarking Base Model")
        print("="*80)
        base_results = pipeline.run_benchmark(
            model_path=args.base_model,
            output_file=str(base_results_file),
            num_samples=args.num_samples
        )

        # Benchmark finetuned model
        trained_results_file = pipeline.results_dir / f"finetuned_model_{pipeline.timestamp}.json"
        print("\n" + "="*80)
        print("STEP 2/3: Benchmarking Finetuned Model")
        print("="*80)
        trained_results = pipeline.run_benchmark(
            model_path=args.finetuned_model,
            output_file=str(trained_results_file),
            num_samples=args.num_samples
        )

        # Compare results
        print("\n" + "="*80)
        print("STEP 3/3: Comparing Results")
        print("="*80)
        comparison = pipeline.compare_results(base_results, trained_results)

    elif args.command == 'benchmark':
        # Benchmark single model
        if args.output_file is None:
            args.output_file = pipeline.results_dir / f"benchmark_{pipeline.timestamp}.json"

        results = pipeline.run_benchmark(
            model_path=args.model_path,
            output_file=str(args.output_file),
            num_samples=args.num_samples
        )

        # Display results
        accuracies = pipeline.calculate_accuracy(results)
        print(f"\nBenchmark Results for {args.model_path}:")
        for benchmark, acc in accuracies.items():
            print(f"  {benchmark}: {acc['correct']}/{acc['total']} = {acc['accuracy']:.1f}%")

    else:
        # Default: run full pipeline (backward compatibility)
        if not hasattr(args, 'num_samples'):
            args.num_samples = 100
        if not hasattr(args, 'skip_training'):
            args.skip_training = False
        if not hasattr(args, 'train_only'):
            args.train_only = False

        pipeline.run_full_pipeline(
            num_samples=args.num_samples,
            train=not args.skip_training,
            skip_base=args.train_only
        )


if __name__ == "__main__":
    main()
