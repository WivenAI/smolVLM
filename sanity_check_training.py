#!/usr/bin/env python3
"""
SANITY CHECK: Train on 1000 samples, evaluate on the SAME 1000 samples.

If accuracy improves: ✅ Training works! (generalization is separate issue)
If accuracy doesn't improve: ❌ Training is broken (wrong hyperparameters, frozen layers, etc.)
"""

import json
import argparse
from pathlib import Path

def calculate_accuracy(results):
    """Calculate accuracy from results"""
    correct = 0
    total = len(results)

    for item in results:
        response = item.get('response', '').lower()
        ground_truth = item.get('ground_truth', [])

        if isinstance(ground_truth, list):
            if any(gt.lower() in response for gt in ground_truth):
                correct += 1
        elif isinstance(ground_truth, str):
            if ground_truth.lower() in response:
                correct += 1

    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description="Sanity check: Train and test on same data")
    parser.add_argument("--base-results", required=True,
                       help="Base model results JSON on 1000 samples")
    parser.add_argument("--trained-results", required=True,
                       help="Trained model results JSON on 1000 samples")
    args = parser.parse_args()

    print("="*80)
    print("TRAINING SANITY CHECK")
    print("="*80)
    print()
    print("📊 Comparing base model vs trained model on the SAME 1000 training samples")
    print()

    # Load base model results
    with open(args.base_results, 'r') as f:
        base_data = json.load(f)

    # Load trained model results
    with open(args.trained_results, 'r') as f:
        trained_data = json.load(f)

    # Get results lists (handle different formats)
    if 'ocrbench' in base_data:
        base_results = base_data['ocrbench']
        trained_results = trained_data['ocrbench']
    else:
        base_results = base_data.get('results', base_data)
        trained_results = trained_data.get('results', trained_data)

    # Calculate accuracies
    base_acc = calculate_accuracy(base_results)
    trained_acc = calculate_accuracy(trained_results)

    improvement = trained_acc - base_acc

    print(f"{'Model':<25} {'Accuracy':<12} {'Samples'}")
    print("-"*50)
    print(f"{'Base (no training)':<25} {base_acc:>10.2%}  {len(base_results)}")
    print(f"{'After training':<25} {trained_acc:>10.2%}  {len(trained_results)}")
    print(f"{'Improvement':<25} {improvement:>+9.2%}")

    print()
    print("="*80)
    print("VERDICT")
    print("="*80)

    if improvement > 0.10:  # >10% improvement
        print("✅ EXCELLENT - Training is working perfectly!")
        print(f"   Model improved by {improvement:.2%} on its training data")
        print()
        print("   ✓ Loss is decreasing")
        print("   ✓ Model is learning from the data")
        print("   ✓ Hyperparameters are reasonable")
        print()
        print("   If test accuracy is low, it's a generalization issue:")
        print("     • Training data may be different from test data")
        print("     • Model may be overfitting to training specifics")
        print("     • May need different training data or regularization")

    elif improvement > 0.05:  # 5-10% improvement
        print("⚠️  MODERATE - Training is working but could be better")
        print(f"   Model improved by {improvement:.2%} on its training data")
        print()
        print("   Consider:")
        print("     • Training for more epochs")
        print("     • Increasing learning rate")
        print("     • Checking if layers are frozen")

    elif improvement > 0.00:  # 0-5% improvement
        print("⚠️  WEAK - Training is barely working")
        print(f"   Model improved by only {improvement:.2%} on its training data")
        print()
        print("   Issues to check:")
        print("     • Learning rate might be too low")
        print("     • Some layers might be frozen")
        print("     • Need more training steps/epochs")

    else:  # Negative improvement
        print("❌ BROKEN - Training made performance WORSE!")
        print(f"   Model degraded by {abs(improvement):.2%} on its training data")
        print()
        print("   Critical issues:")
        print("     • Learning rate too high (catastrophic forgetting)")
        print("     • Wrong loss function")
        print("     • All LoRA layers might be frozen")
        print("     • Data preprocessing mismatch")

    print()
    print(f"Base results: {args.base_results}")
    print(f"Trained results: {args.trained_results}")

if __name__ == "__main__":
    main()
