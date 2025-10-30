#!/usr/bin/env python3
"""Test that the accuracy calculation fix works correctly"""

import json
from pathlib import Path

# Import the fixed pipeline
from run_systematic_benchmark_pipeline import SystematicBenchmarkPipeline
import argparse

# Create a dummy args object
args = argparse.Namespace(
    benchmark_percentage=100.0,
    num_samples=None,
    test_mode=False,
    no_wandb=True,
    continue_on_error=False,
    benchmarks=None,
    skip_baseline=False,
    train_benchmark=None,
    skip_benchmark_training=False,
    train_samples=500,
    epochs=3,
    train_erp=False,
    skip_erp_training=False,
    erp_strategy='qcm',
    qcm_dataset='',
    dpo_dataset='',
    image_dir=''
)

pipeline = SystematicBenchmarkPipeline(args)

# Test with actual results from most recent base model evaluation
result_file = Path("systematic_results/base_model_20251030_092630.json")

if result_file.exists():
    print("Testing accuracy calculation on real data...")
    print("="*80)

    with open(result_file, 'r') as f:
        results = json.load(f)

    for benchmark_name, benchmark_results in results.items():
        if isinstance(benchmark_results, list) and benchmark_results:
            # Calculate with OLD broken method (without benchmark_name)
            old_accuracy = 0.0
            correct = 0
            total = 0
            for result in benchmark_results:
                if 'ground_truth' in result and 'response' in result:
                    gt = str(result['ground_truth']).lower().strip()
                    response = str(result['response']).lower().strip()
                    if gt in response or response in gt:
                        correct += 1
                    total += 1
            old_accuracy = (correct / total * 100) if total > 0 else 0.0

            # Calculate with NEW fixed method
            new_accuracy = pipeline.calculate_accuracy(benchmark_results, benchmark_name=benchmark_name)

            print(f"\n{benchmark_name.upper()}:")
            print(f"  Samples: {len(benchmark_results)}")
            print(f"  OLD (broken) accuracy: {old_accuracy:.2f}%")
            print(f"  NEW (fixed) accuracy:  {new_accuracy:.2f}%")
            print(f"  Difference: {new_accuracy - old_accuracy:+.2f}%")

            if new_accuracy > 40:
                print(f"  ✅ Fixed! Now showing realistic accuracy")
            elif new_accuracy > old_accuracy:
                print(f"  ✅ Improvement, but may need more tuning")
            else:
                print(f"  ⚠️  No improvement")

    print("\n" + "="*80)
    print("Summary:")
    print("  If NEW accuracies are 40-60%, the fix is working!")
    print("  If they're still 3-7%, something else is wrong.")
    print("="*80)
else:
    print(f"❌ Could not find test file: {result_file}")
    print("Run evaluation first to generate test data")
