#!/usr/bin/env python3
"""
Simple proof that training works:
1. Evaluate BASE model on 1000 samples
2. Train on those SAME 1000 samples
3. Evaluate TRAINED model on those SAME 1000 samples
4. Compare to prove training improves performance
"""

import subprocess
import json
import sys

def run_command(cmd, description):
    """Run a command and show output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        sys.exit(1)

    print(f"\n✅ {description} completed!")
    return result

def main():
    SAMPLES = 1000
    SEED = 42
    TRAINED_MODEL = "proof_trained_1000"

    print(f"\n{'#'*60}")
    print(f"PROOF THAT TRAINING WORKS")
    print(f"{'#'*60}")
    print(f"Samples: {SAMPLES} (same for training AND evaluation)")
    print(f"Seed: {SEED}")
    print(f"{'#'*60}\n")

    # Step 1: Evaluate BASE model
    print("\nSTEP 1: Evaluate BASE model on 1000 samples")
    run_command([
        "python3", "sanity_check_eval.py",
        "--model-path", "HuggingFaceTB/SmolVLM-500M-Instruct",
        "--benchmark", "ocrbench",
        "--max-samples", str(SAMPLES),
        "--seed", str(SEED),
        "--output", "proof_base_model.json"
    ], "Base Model Evaluation")

    # Step 2: Train on those SAME 1000 samples
    print("\n\nSTEP 2: Train on those SAME 1000 samples")
    run_command([
        "python3", "finetune_on_benchmarks.py",
        "--benchmark", "ocrbench",
        "--num-epochs", "1",
        "--max-samples", str(SAMPLES),
        "--output-dir", TRAINED_MODEL
    ], "Training")

    # Step 3: Evaluate TRAINED model on SAME 1000 samples
    print("\n\nSTEP 3: Evaluate TRAINED model on SAME 1000 samples")
    run_command([
        "python3", "sanity_check_eval.py",
        "--model-path", TRAINED_MODEL,
        "--benchmark", "ocrbench",
        "--max-samples", str(SAMPLES),
        "--seed", str(SEED),
        "--output", "proof_trained_model.json"
    ], "Trained Model Evaluation")

    # Step 4: Compare results
    print("\n\nSTEP 4: Compare results")

    with open("proof_base_model.json", 'r') as f:
        base_data = json.load(f)

    with open("proof_trained_model.json", 'r') as f:
        trained_data = json.load(f)

    base_acc = base_data["accuracy"]
    trained_acc = trained_data["accuracy"]
    improvement = trained_acc - base_acc

    print(f"\n{'#'*60}")
    print(f"FINAL RESULTS - PROOF THAT TRAINING WORKS")
    print(f"{'#'*60}")
    print(f"Same {SAMPLES} samples used for training AND evaluation")
    print(f"\nBASE MODEL accuracy:    {base_acc:.2f}%")
    print(f"TRAINED MODEL accuracy: {trained_acc:.2f}%")
    print(f"\nImprovement: {improvement:+.2f}%")

    if trained_acc > base_acc + 10:
        print(f"\n✅✅✅ SUCCESS! Training WORKS!")
        print(f"   Trained model is {improvement:.2f}% better!")
        print(f"   This proves the training mechanism is functional.")
    elif trained_acc > base_acc:
        print(f"\n✅ Training shows improvement (+{improvement:.2f}%)")
    else:
        print(f"\n❌ WARNING: Training did not improve the model")

    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
