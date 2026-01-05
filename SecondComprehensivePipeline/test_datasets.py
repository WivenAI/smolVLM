#!/usr/bin/env python3
"""
Unit tests to validate that all datasets referenced in config are present and ready.

This test suite verifies:
- Dataset JSON files exist and are valid
- Image directories exist
- Sample images from datasets are accessible
- HuggingFace benchmark datasets can be loaded

Usage:
    python test_datasets.py
    python test_datasets.py --config config/conf.yaml
    python test_datasets.py --verbose
"""

import unittest
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys


class TestDatasets(unittest.TestCase):
    """Test suite for dataset validation"""

    @classmethod
    def setUpClass(cls):
        """Load config once for all tests"""
        cls.base_path = Path(__file__).parent
        cls.config_path = cls.base_path / "config" / "conf.yaml"

        # Allow override from command line
        if hasattr(sys, '_test_config_path'):
            cls.config_path = Path(sys._test_config_path)

        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)

        print(f"\n{'='*80}")
        print(f"Testing datasets from config: {cls.config_path}")
        print(f"{'='*80}\n")

    def _check_dataset_file(self, dataset_path: str, image_dir: str = None) -> Tuple[bool, str, Dict]:
        """Check if dataset file exists and is valid JSON"""
        full_path = self.base_path / dataset_path

        if not full_path.exists():
            return False, f"Dataset file not found: {full_path}", {}

        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in {full_path}: {e}", {}

        if not isinstance(data, list):
            return False, f"Dataset should be a list, got {type(data)}", {}

        if len(data) == 0:
            return False, f"Dataset is empty: {full_path}", {}

        return True, f"Valid dataset with {len(data)} items", {"count": len(data), "data": data}

    def _check_image_dir(self, image_dir: str) -> Tuple[bool, str]:
        """Check if image directory exists"""
        full_path = self.base_path / image_dir

        if not full_path.exists():
            return False, f"Image directory not found: {full_path}"

        if not full_path.is_dir():
            return False, f"Path is not a directory: {full_path}"

        return True, f"Image directory exists: {full_path}"

    def _check_sample_images(self, dataset_path: str, image_dir: str, num_samples: int = 5) -> Tuple[bool, str]:
        """Check if sample images from dataset exist"""
        full_dataset_path = self.base_path / dataset_path
        full_image_dir = self.base_path / image_dir

        try:
            with open(full_dataset_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return False, f"Could not read dataset: {e}"

        # Check first few samples
        samples_to_check = min(num_samples, len(data))
        missing_images = []

        for i, item in enumerate(data[:samples_to_check]):
            image_path = item.get("image_path", "")
            if image_path:
                full_image_path = full_image_dir / image_path
                if not full_image_path.exists():
                    missing_images.append(str(image_path))

        if missing_images:
            return False, f"Missing {len(missing_images)}/{samples_to_check} sample images: {missing_images[:3]}"

        return True, f"All {samples_to_check} sample images exist"

    def test_training_strategy_datasets(self):
        """Test all datasets referenced in training strategies"""
        strategies = self.config.get("training", {}).get("strategies", [])
        enabled_strategies = [s for s in strategies if s.get("enabled", True)]

        print(f"\nTesting {len(enabled_strategies)} enabled training strategies...")

        for strategy in enabled_strategies:
            name = strategy["name"]
            strategy_type = strategy["type"]

            # Skip baseline (no dataset)
            if strategy_type == "none":
                continue

            with self.subTest(strategy=name):
                # Check single dataset strategies
                if "dataset" in strategy:
                    dataset_path = strategy["dataset"]
                    image_dir = strategy.get("image_dir")

                    # Check dataset file
                    success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                    self.assertTrue(success, f"[{name}] {msg}")
                    print(f"  ✓ {name}: Dataset file OK")

                    # Check image directory if specified
                    if image_dir:
                        success, msg = self._check_image_dir(image_dir)
                        self.assertTrue(success, f"[{name}] {msg}")
                        print(f"  ✓ {name}: Image directory OK")

                        # Check sample images
                        success, msg = self._check_sample_images(dataset_path, image_dir)
                        self.assertTrue(success, f"[{name}] {msg}")
                        print(f"  ✓ {name}: Sample images OK")

                # Check multi-dataset strategies (combined)
                if "datasets" in strategy:
                    for dataset_path in strategy["datasets"]:
                        image_dir = strategy.get("image_dir")

                        success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                        self.assertTrue(success, f"[{name}] {msg}")
                        print(f"  ✓ {name}: Dataset {Path(dataset_path).name} OK")

                        if image_dir:
                            success, msg = self._check_sample_images(dataset_path, image_dir)
                            self.assertTrue(success, f"[{name}] {msg}")

                # Check QCM + DPO strategies
                if "qcm_dataset" in strategy:
                    dataset_path = strategy["qcm_dataset"]
                    image_dir = strategy.get("image_dir")

                    success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                    self.assertTrue(success, f"[{name}] QCM dataset: {msg}")
                    print(f"  ✓ {name}: QCM dataset OK")

                if "dpo_dataset" in strategy:
                    dataset_path = strategy["dpo_dataset"]
                    image_dir = strategy.get("image_dir")

                    success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                    self.assertTrue(success, f"[{name}] DPO dataset: {msg}")
                    print(f"  ✓ {name}: DPO dataset OK")

                if "chosen_rej_dataset" in strategy:
                    dataset_path = strategy["chosen_rej_dataset"]
                    image_dir = strategy.get("image_dir")

                    success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                    self.assertTrue(success, f"[{name}] Chosen/Rejected dataset: {msg}")
                    print(f"  ✓ {name}: Chosen/Rejected dataset OK")

                # Check multi-dataset fields
                if "qcm_datasets" in strategy:
                    for dataset_path in strategy["qcm_datasets"]:
                        success, msg, _ = self._check_dataset_file(dataset_path)
                        self.assertTrue(success, f"[{name}] QCM dataset {Path(dataset_path).name}: {msg}")

                if "chosen_rej_datasets" in strategy:
                    for dataset_path in strategy["chosen_rej_datasets"]:
                        success, msg, _ = self._check_dataset_file(dataset_path)
                        self.assertTrue(success, f"[{name}] Chosen/Rej dataset {Path(dataset_path).name}: {msg}")

    def test_evaluation_erp_datasets(self):
        """Test all ERP evaluation datasets"""
        erp_eval = self.config.get("evaluation", {}).get("erp_evaluation", {})
        enabled_evals = {k: v for k, v in erp_eval.items()
                        if isinstance(v, dict) and v.get("enabled", True)}

        print(f"\nTesting {len(enabled_evals)} enabled ERP evaluation datasets...")

        for eval_name, eval_config in enabled_evals.items():
            with self.subTest(evaluation=eval_name):
                if "dataset" in eval_config:
                    dataset_path = eval_config["dataset"]
                    image_dir = eval_config.get("image_dir")

                    # Check dataset file
                    success, msg, _ = self._check_dataset_file(dataset_path, image_dir)
                    self.assertTrue(success, f"[{eval_name}] {msg}")
                    print(f"  ✓ {eval_name}: Dataset file OK")

                    # Check image directory if specified
                    if image_dir:
                        success, msg = self._check_image_dir(image_dir)
                        self.assertTrue(success, f"[{eval_name}] {msg}")
                        print(f"  ✓ {eval_name}: Image directory OK")

                        # Check sample images
                        success, msg = self._check_sample_images(dataset_path, image_dir, num_samples=3)
                        self.assertTrue(success, f"[{eval_name}] {msg}")
                        print(f"  ✓ {eval_name}: Sample images OK")

    def test_benchmark_datasets(self):
        """Test HuggingFace benchmark datasets"""
        benchmarks = self.config.get("evaluation", {}).get("benchmarks", [])
        enabled_benchmarks = [b for b in benchmarks if b.get("enabled", True)]

        if not enabled_benchmarks:
            print("\nNo enabled benchmarks to test")
            return

        print(f"\nTesting {len(enabled_benchmarks)} enabled HuggingFace benchmarks...")
        print("(Note: This will attempt to load datasets, may take time on first run)")

        for benchmark in enabled_benchmarks:
            name = benchmark["name"]
            dataset_name = benchmark["dataset"]
            split = benchmark.get("split", "test")

            with self.subTest(benchmark=name):
                try:
                    # Try to import datasets library
                    from datasets import load_dataset

                    # Try to load the dataset (may download on first run)
                    print(f"  → Loading {dataset_name} ({split})...")
                    dataset = load_dataset(dataset_name, split=split)

                    self.assertIsNotNone(dataset, f"[{name}] Failed to load dataset")
                    self.assertGreater(len(dataset), 0, f"[{name}] Dataset is empty")

                    print(f"  ✓ {name}: Dataset loaded ({len(dataset)} samples)")

                except ImportError:
                    self.skipTest("datasets library not installed")
                except Exception as e:
                    self.fail(f"[{name}] Failed to load benchmark dataset: {e}")

    def test_procedure_qcm_answer_distribution(self):
        """Test that procedure QCM datasets have balanced answer distribution (max 28% per answer)"""
        print("\nTesting procedure QCM answer distribution...")

        # Find procedure QCM datasets
        procedure_datasets = []

        # Check training strategies
        strategies = self.config.get("training", {}).get("strategies", [])
        enabled_strategies = [s for s in strategies if s.get("enabled", True)]

        for strategy in enabled_strategies:
            # Check for procedure datasets in various fields
            datasets_to_check = []

            if "dataset" in strategy:
                if "procedure" in strategy["dataset"].lower():
                    datasets_to_check.append((strategy["name"], strategy["dataset"]))

            if "qcm_dataset" in strategy:
                if "procedure" in strategy["qcm_dataset"].lower():
                    datasets_to_check.append((strategy["name"], strategy["qcm_dataset"]))

            if "qcm_datasets" in strategy:
                for ds in strategy["qcm_datasets"]:
                    if "procedure" in ds.lower():
                        datasets_to_check.append((strategy["name"], ds))

            for strategy_name, dataset_path in datasets_to_check:
                if dataset_path not in [d[1] for d in procedure_datasets]:
                    procedure_datasets.append((strategy_name, dataset_path))

        # Check evaluation datasets
        erp_eval = self.config.get("evaluation", {}).get("erp_evaluation", {})
        for eval_name, eval_config in erp_eval.items():
            if isinstance(eval_config, dict) and eval_config.get("enabled", True):
                if "dataset" in eval_config and "procedure" in eval_config["dataset"].lower():
                    dataset_path = eval_config["dataset"]
                    if dataset_path not in [d[1] for d in procedure_datasets]:
                        procedure_datasets.append((eval_name, dataset_path))

        if not procedure_datasets:
            print("  No procedure QCM datasets found")
            return

        print(f"  Found {len(procedure_datasets)} procedure QCM datasets")

        # Check each procedure dataset
        for dataset_name, dataset_path in procedure_datasets:
            with self.subTest(dataset=dataset_name):
                full_path = self.base_path / dataset_path

                # Load dataset
                with open(full_path, 'r') as f:
                    data = json.load(f)

                # Count answer distribution
                answer_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
                total = len(data)

                for item in data:
                    # Handle nested structure (procedure datasets have qcm.correct_answer)
                    correct_answer = None
                    if "correct_answer" in item:
                        correct_answer = item["correct_answer"]
                    elif "qcm" in item and "correct_answer" in item["qcm"]:
                        correct_answer = item["qcm"]["correct_answer"]

                    if correct_answer:
                        correct_answer = correct_answer.strip().upper()
                        if correct_answer in answer_counts:
                            answer_counts[correct_answer] += 1

                # Calculate percentages
                answer_percentages = {k: (v / total * 100) if total > 0 else 0
                                     for k, v in answer_counts.items()}

                # Check that no answer exceeds 28%
                print(f"\n  {dataset_name} ({Path(dataset_path).name}):")
                print(f"    Total questions: {total}")
                for answer in sorted(answer_counts.keys()):
                    count = answer_counts[answer]
                    pct = answer_percentages[answer]
                    status = "✓" if pct <= 28 else "✗"
                    print(f"    {status} {answer}: {count:4d} ({pct:5.2f}%)")

                # Assert all answers are <= 28%
                for answer, pct in answer_percentages.items():
                    self.assertLessEqual(
                        pct, 28.0,
                        f"[{dataset_name}] Answer '{answer}' appears {pct:.2f}% of the time (max allowed: 28%)"
                    )

                print(f"  ✓ {dataset_name}: Answer distribution is balanced")

    def test_dataset_statistics(self):
        """Print statistics about datasets"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)

        strategies = self.config.get("training", {}).get("strategies", [])
        enabled_strategies = [s for s in strategies if s.get("enabled", True)]

        # Collect all unique datasets
        dataset_info = {}

        for strategy in enabled_strategies:
            datasets_to_check = []

            if "dataset" in strategy:
                datasets_to_check.append(strategy["dataset"])
            if "datasets" in strategy:
                datasets_to_check.extend(strategy["datasets"])
            if "qcm_dataset" in strategy:
                datasets_to_check.append(strategy["qcm_dataset"])
            if "dpo_dataset" in strategy:
                datasets_to_check.append(strategy["dpo_dataset"])
            if "chosen_rej_dataset" in strategy:
                datasets_to_check.append(strategy["chosen_rej_dataset"])
            if "qcm_datasets" in strategy:
                datasets_to_check.extend(strategy["qcm_datasets"])
            if "chosen_rej_datasets" in strategy:
                datasets_to_check.extend(strategy["chosen_rej_datasets"])

            for dataset_path in datasets_to_check:
                if dataset_path not in dataset_info:
                    success, msg, data = self._check_dataset_file(dataset_path)
                    if success:
                        dataset_info[dataset_path] = data["count"]

        # Print summary
        print("\nLocal JSON Datasets:")
        for dataset_path, count in sorted(dataset_info.items()):
            print(f"  {Path(dataset_path).name:50s} {count:6d} samples")

        print(f"\nTotal unique datasets: {len(dataset_info)}")
        print(f"Total samples across all datasets: {sum(dataset_info.values())}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test dataset availability")
    parser.add_argument("--config", type=str, default="config/conf.yaml",
                       help="Path to config file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Store config path for tests
    sys._test_config_path = args.config

    # Run tests
    if args.verbose:
        verbosity = 2
    else:
        verbosity = 1

    # Run unittest
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDatasets)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
