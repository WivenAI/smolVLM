#!/usr/bin/env python3
"""
Evaluate model on its own TRAINING set to verify training is working.
High accuracy on train set = training works, low test accuracy = generalization issue.
"""

import sys
import json
import argparse
from pathlib import Path
from evaluate_ocrbench import OCRBenchEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--train-data", default="test_ocrbench_1epoch/train_samples.json",
                       help="Path to training samples JSON")
    parser.add_argument("--output", default="train_set_evaluation.json")
    args = parser.parse_args()

    # First, we need to save the training samples
    # Let's check if they exist
    train_samples_path = Path(args.train_data)

    if not train_samples_path.exists():
        print(f"❌ Training samples not found at {train_samples_path}")
        print("   Need to save training samples during training first")
        sys.exit(1)

    # Load training samples
    with open(train_samples_path, 'r') as f:
        train_samples = json.load(f)

    print(f"="*80)
    print(f"Train Set Evaluation")
    print(f"="*80)
    print(f"Model: {args.model_path}")
    print(f"Train samples: {len(train_samples)}")
    print()

    # Initialize evaluator
    evaluator = OCRBenchEvaluator(
        model_name=args.model_path,
        output_dir=Path(args.output).parent,
        dataset_percentage=100
    )

    # Evaluate on training samples
    print("Evaluating on TRAINING SET...")
    results = []

    for idx, sample in enumerate(train_samples):
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(train_samples)}")

        # Generate response
        response = evaluator.model.generate_response(
            sample['image'],
            sample['question']
        )

        results.append({
            'question': sample['question'],
            'response': response,
            'ground_truth': sample['ground_truth'],
            'task_type': sample.get('task_type', 'unknown'),
            'dataset': sample.get('dataset', 'unknown')
        })

    # Calculate accuracy
    correct = 0
    for r in results:
        gt = r['ground_truth']
        resp = r['response'].lower()

        if isinstance(gt, list):
            if any(g.lower() in resp for g in gt):
                correct += 1
        elif isinstance(gt, str):
            if gt.lower() in resp:
                correct += 1

    accuracy = correct / len(results) if results else 0

    # Save results
    output_data = {
        'model': args.model_path,
        'num_samples': len(results),
        'accuracy': accuracy,
        'results': results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n" + "="*80)
    print(f"TRAIN SET PERFORMANCE")
    print(f"="*80)
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"\n💡 INTERPRETATION:")
    if accuracy > 0.90:
        print(f"   ✅ EXCELLENT - Training is working! Model learned the training data.")
        print(f"      If test accuracy is low, it's a generalization problem, not training.")
    elif accuracy > 0.70:
        print(f"   ⚠️  MODERATE - Training partially working, may need more epochs/tuning")
    else:
        print(f"   ❌ POOR - Training is NOT working properly!")
        print(f"      Check: learning rate, loss function, data format, model freezing")

    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()
