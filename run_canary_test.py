#!/usr/bin/env python3
"""
Canary Test Pipeline - Verify that training actually works

This script:
1. Creates a small canary dataset (5 simple, memorizable examples)
2. Trains model on canary dataset (both QCM and DPO)
3. Evaluates on the SAME canary dataset
4. Checks if model achieves near-perfect accuracy (>90%)

If the model doesn't memorize the canary dataset, there's a bug in:
- Training code
- Data loading
- Model architecture
- Optimization setup

This is a critical sanity check before running full training.
"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from tqdm import tqdm


class CanaryTestPipeline:
    """Run canary test to verify training works"""

    def __init__(self, args):
        self.args = args
        self.canary_dir = Path("canary_dataset")
        self.canary_qcm_model = Path("./canary_qcm_model")
        self.canary_dpo_model = Path("./canary_dpo_model")
        self.results = {}

    def run_command(self, cmd: list, description: str):
        """Run a command and handle errors"""
        print("\n" + "="*80)
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print("="*80 + "\n")

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"❌ Error running {description}")
            return False

        print(f"✅ Completed: {description}")
        return True

    def create_canary_dataset(self):
        """Step 1: Create canary dataset"""
        print("\n" + "🐤 "*20)
        print("STEP 1: Creating Canary Dataset")
        print("🐤 "*20)

        if self.canary_dir.exists() and not self.args.recreate:
            print(f"Canary dataset already exists at {self.canary_dir}")
            print("Use --recreate to recreate it")
            return True

        cmd = [
            "python3", "create_canary_dataset.py",
            "--output-dir", str(self.canary_dir),
            "--num-samples", str(self.args.num_samples),
            "--type", "both"
        ]

        return self.run_command(cmd, "Create canary dataset")

    def train_canary_qcm(self):
        """Step 2a: Train on canary QCM dataset"""
        print("\n" + "🎓 "*20)
        print("STEP 2a: Training on Canary QCM Dataset")
        print("🎓 "*20)

        cmd = [
            "python3", "finetune_smolvlm_qcm.py",
            "--base-model", "HuggingFaceTB/SmolVLM-500M-Instruct",
            "--output-dir", str(self.canary_qcm_model),
            "--dataset", str(self.canary_dir / "canary_qcm.json"),
            "--image-dir", str(self.canary_dir),
            "--num-epochs", str(self.args.epochs)
        ]

        return self.run_command(cmd, "Train on canary QCM")

    def train_canary_dpo(self):
        """Step 2b: Train on canary DPO dataset"""
        print("\n" + "🎓 "*20)
        print("STEP 2b: Training on Canary DPO Dataset")
        print("🎓 "*20)

        cmd = [
            "python3", "finetune_smolvlm_dpo.py",
            "--base-model", "HuggingFaceTB/SmolVLM-500M-Instruct",
            "--output-dir", str(self.canary_dpo_model),
            "--dataset", str(self.canary_dir / "canary_dpo.json"),
            "--image-dir", str(self.canary_dir)
        ]

        return self.run_command(cmd, "Train on canary DPO")

    def evaluate_canary_qcm(self):
        """Step 3a: Evaluate on canary QCM dataset"""
        print("\n" + "📊 "*20)
        print("STEP 3a: Evaluating Canary QCM Model")
        print("📊 "*20)

        # Load model
        print(f"Loading model from {self.canary_qcm_model}...")
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.canary_qcm_model),
                trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                str(self.canary_qcm_model),
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

        # Load canary dataset
        canary_path = self.canary_dir / "canary_qcm.json"
        with open(canary_path, 'r') as f:
            canary_data = json.load(f)

        print(f"\nEvaluating on {len(canary_data)} canary examples...")

        correct = 0
        total = len(canary_data)
        results = []

        for item in tqdm(canary_data, desc="Canary QCM Evaluation"):
            # Load image
            image_path = self.canary_dir / item['image_name']
            image = Image.open(image_path).convert('RGB')

            # Format question
            qcm = item['qcm']
            options_text = "\n".join([f"{k}: {v}" for k, v in qcm['options'].items()])
            question = f"{qcm['question']}\n\nOptions:\n{options_text}\n\nAnswer:"

            # Generate response
            text = f"<image>{question}"
            inputs = processor(
                images=image,
                text=text,
                return_tensors="pt",
                size={"longest_edge": 1024}
            ).to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=50)
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.replace(question, "").strip()

            # Check if correct
            correct_answer = qcm['correct_answer']
            is_correct = correct_answer in response.upper()

            if is_correct:
                correct += 1

            results.append({
                "image": item['image_name'],
                "question": qcm['question'],
                "correct_answer": correct_answer,
                "model_response": response,
                "is_correct": is_correct
            })

        accuracy = (correct / total * 100) if total > 0 else 0.0

        self.results['qcm'] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results
        }

        print(f"\n{'='*80}")
        print(f"QCM Canary Test Results: {correct}/{total} correct ({accuracy:.1f}%)")
        print(f"{'='*80}\n")

        return True

    def evaluate_canary_dpo(self):
        """Step 3b: Evaluate on canary DPO dataset"""
        print("\n" + "📊 "*20)
        print("STEP 3b: Evaluating Canary DPO Model")
        print("📊 "*20)

        # Load model
        print(f"Loading model from {self.canary_dpo_model}...")
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.canary_dpo_model),
                trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                str(self.canary_dpo_model),
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

        # Load canary dataset
        canary_path = self.canary_dir / "canary_dpo.json"
        with open(canary_path, 'r') as f:
            canary_data = json.load(f)

        print(f"\nEvaluating on {len(canary_data)} canary examples...")

        correct = 0
        total = len(canary_data)
        results = []

        for item in tqdm(canary_data, desc="Canary DPO Evaluation"):
            # Load image
            image_path = self.canary_dir / item['image_name']
            image = Image.open(image_path).convert('RGB')

            # Generate response
            text = f"<image>{item['prompt']}"
            inputs = processor(
                images=image,
                text=text,
                return_tensors="pt",
                size={"longest_edge": 1024}
            ).to(device)

            generated_ids = model.generate(**inputs, max_new_tokens=100)
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.replace(item['prompt'], "").strip()

            # Check if response is more similar to chosen than rejected
            chosen = item['chosen'].lower()
            rejected = item['rejected'].lower()
            response_lower = response.lower()

            # Simple similarity check
            chosen_words = set(chosen.split())
            rejected_words = set(rejected.split())
            response_words = set(response_lower.split())

            chosen_overlap = len(response_words & chosen_words)
            rejected_overlap = len(response_words & rejected_words)

            is_correct = chosen_overlap > rejected_overlap

            if is_correct:
                correct += 1

            results.append({
                "image": item['image_name'],
                "prompt": item['prompt'],
                "chosen": item['chosen'],
                "rejected": item['rejected'],
                "model_response": response,
                "is_correct": is_correct
            })

        accuracy = (correct / total * 100) if total > 0 else 0.0

        self.results['dpo'] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results
        }

        print(f"\n{'='*80}")
        print(f"DPO Canary Test Results: {correct}/{total} correct ({accuracy:.1f}%)")
        print(f"{'='*80}\n")

        return True

    def generate_report(self):
        """Step 4: Generate final report"""
        print("\n" + "="*80)
        print("CANARY TEST REPORT")
        print("="*80)

        # Save results to file
        report_path = Path("canary_test_results.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nDetailed results saved to: {report_path}")

        # Print summary
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)

        all_passed = True

        if 'qcm' in self.results:
            qcm_acc = self.results['qcm']['accuracy']
            qcm_passed = qcm_acc >= self.args.threshold
            status = "✅ PASS" if qcm_passed else "❌ FAIL"
            print(f"\nQCM Canary Test: {status}")
            print(f"  Accuracy: {qcm_acc:.1f}% (threshold: {self.args.threshold}%)")
            print(f"  Result: {self.results['qcm']['correct']}/{self.results['qcm']['total']}")

            if not qcm_passed:
                all_passed = False
                print("  ⚠️  Model failed to memorize QCM canary dataset!")
                print("  ⚠️  This indicates a problem with QCM training code.")

        if 'dpo' in self.results:
            dpo_acc = self.results['dpo']['accuracy']
            dpo_passed = dpo_acc >= self.args.threshold
            status = "✅ PASS" if dpo_passed else "❌ FAIL"
            print(f"\nDPO Canary Test: {status}")
            print(f"  Accuracy: {dpo_acc:.1f}% (threshold: {self.args.threshold}%)")
            print(f"  Result: {self.results['dpo']['correct']}/{self.results['dpo']['total']}")

            if not dpo_passed:
                all_passed = False
                print("  ⚠️  Model failed to memorize DPO canary dataset!")
                print("  ⚠️  This indicates a problem with DPO training code.")

        print("\n" + "="*80)
        if all_passed:
            print("🎉 ALL CANARY TESTS PASSED - Training code works correctly!")
        else:
            print("❌ CANARY TESTS FAILED - There are bugs in training code!")
            print("   Fix these issues before running full training.")
        print("="*80 + "\n")

        return all_passed

    def run_full_pipeline(self):
        """Run complete canary test pipeline"""
        print("\n" + "🐤 "*20)
        print("CANARY TEST PIPELINE")
        print("This will verify that training actually works by:")
        print("  1. Creating tiny, memorizable dataset")
        print("  2. Training model on it")
        print("  3. Testing if model memorized it (should be ~100%)")
        print("🐤 "*20)

        # Step 1: Create canary dataset
        if not self.create_canary_dataset():
            print("❌ Failed to create canary dataset")
            return False

        # Step 2a: Train QCM
        if not self.args.skip_qcm:
            if not self.train_canary_qcm():
                print("❌ Failed to train canary QCM model")
                return False

            # Step 3a: Evaluate QCM
            if not self.evaluate_canary_qcm():
                print("❌ Failed to evaluate canary QCM model")
                return False

        # Step 2b: Train DPO
        if not self.args.skip_dpo:
            if not self.train_canary_dpo():
                print("❌ Failed to train canary DPO model")
                return False

            # Step 3b: Evaluate DPO
            if not self.evaluate_canary_dpo():
                print("❌ Failed to evaluate canary DPO model")
                return False

        # Step 4: Generate report
        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(
        description="Run canary test to verify training works"
    )

    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of canary samples (default: 5, max: 5)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs for canary (default: 10, should overfit)")
    parser.add_argument("--threshold", type=float, default=80.0,
                       help="Accuracy threshold to pass (default: 80%%)")
    parser.add_argument("--recreate", action="store_true",
                       help="Recreate canary dataset even if exists")
    parser.add_argument("--skip-qcm", action="store_true",
                       help="Skip QCM canary test")
    parser.add_argument("--skip-dpo", action="store_true",
                       help="Skip DPO canary test")

    args = parser.parse_args()

    # Run pipeline
    pipeline = CanaryTestPipeline(args)
    success = pipeline.run_full_pipeline()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
