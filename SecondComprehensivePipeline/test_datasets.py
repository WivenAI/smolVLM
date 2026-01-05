#!/usr/bin/env python3
"""
Simple test to check if all datasets are present and ready.

Usage:
    python test_datasets.py
"""

import unittest
import json
import yaml
from pathlib import Path
import sys


class TestDatasets(unittest.TestCase):
    """Quick dataset availability check"""

    @classmethod
    def setUpClass(cls):
        cls.base_path = Path(__file__).parent
        cls.config_path = cls.base_path / "config" / "conf.yaml"

        with open(cls.config_path, 'r') as f:
            cls.config = yaml.safe_load(f)

    def test_all_datasets_exist(self):
        """Check all dataset files exist"""
        strategies = self.config.get("training", {}).get("strategies", [])
        enabled = [s for s in strategies if s.get("enabled", True)]

        missing = []
        for strategy in enabled:
            # Check all dataset fields
            for field in ["dataset", "qcm_dataset", "dpo_dataset", "chosen_rej_dataset"]:
                if field in strategy:
                    path = self.base_path / strategy[field]
                    if not path.exists():
                        missing.append(f"{strategy['name']}: {strategy[field]}")

            # Check multi-dataset fields
            for field in ["datasets", "qcm_datasets", "chosen_rej_datasets"]:
                if field in strategy:
                    for ds in strategy[field]:
                        path = self.base_path / ds
                        if not path.exists():
                            missing.append(f"{strategy['name']}: {ds}")

        # Check evaluation datasets
        erp = self.config.get("evaluation", {}).get("erp_evaluation", {})
        for name, cfg in erp.items():
            if isinstance(cfg, dict) and cfg.get("enabled") and "dataset" in cfg:
                path = self.base_path / cfg["dataset"]
                if not path.exists():
                    missing.append(f"{name}: {cfg['dataset']}")

        self.assertEqual(len(missing), 0, f"Missing datasets:\n" + "\n".join(missing))

    def test_procedure_qcm_answer_balance(self):
        """Check procedure QCMs have <28% per answer"""
        strategies = self.config.get("training", {}).get("strategies", [])
        erp = self.config.get("evaluation", {}).get("erp_evaluation", {})

        procedure_datasets = []
        for s in strategies:
            if s.get("enabled") and "procedure" in str(s.get("dataset", "")).lower():
                procedure_datasets.append(s["dataset"])
            if s.get("enabled") and "procedure" in str(s.get("qcm_dataset", "")).lower():
                procedure_datasets.append(s["qcm_dataset"])

        for name, cfg in erp.items():
            if isinstance(cfg, dict) and cfg.get("enabled") and "dataset" in cfg:
                if "procedure" in cfg["dataset"].lower():
                    procedure_datasets.append(cfg["dataset"])

        procedure_datasets = list(set(procedure_datasets))

        failed = []
        for ds_path in procedure_datasets:
            with open(self.base_path / ds_path) as f:
                data = json.load(f)

            counts = {"A": 0, "B": 0, "C": 0, "D": 0}
            for item in data:
                ans = item.get("correct_answer") or item.get("qcm", {}).get("correct_answer", "")
                ans = ans.strip().upper()
                if ans in counts:
                    counts[ans] += 1

            total = len(data)
            for letter, count in counts.items():
                pct = (count / total * 100) if total > 0 else 0
                if pct > 28:
                    failed.append(f"{Path(ds_path).name}: {letter}={pct:.1f}%")

        self.assertEqual(len(failed), 0, f"Answer imbalance:\n" + "\n".join(failed))


if __name__ == "__main__":
    unittest.main(verbosity=2)
