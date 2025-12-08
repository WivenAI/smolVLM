"""
SecondComprehensivePipeline - Clean VLM training and evaluation pipeline for reducing hallucinations

This module provides a modular, config-driven approach to:
1. Fine-tuning SmolVLM using SFT (QCM) and DPO methods
2. Evaluating on benchmarks (OCRBench, DocVQA, ChartQA)
3. Evaluating on ERP-specific QCM questions
4. Comparing results across different training strategies

Usage:
    python pipeline.py                    # Run full pipeline
    python pipeline.py --eval-only        # Evaluation only
    python pipeline.py --debug            # Debug mode (10 samples)

Configuration:
    Edit config/conf.yaml to configure training strategies and evaluation settings.

Structure:
    SecondComprehensivePipeline/
    ├── config/              # Configuration files
    │   └── conf.yaml        # Main config
    ├── datasets/            # Dataset storage
    │   ├── cache/           # Cached benchmark data
    │   ├── dpo/             # DPO datasets
    │   ├── qcm/             # QCM datasets
    │   └── images/          # ERP screenshots
    ├── eval/                # Evaluators
    │   ├── base_evaluator.py
    │   ├── evaluator_ocr.py
    │   ├── evaluator_docvqa.py
    │   ├── evaluator_chartqa.py
    │   ├── evaluator_qcm.py
    │   └── evaluator_all.py
    ├── train/               # Trainers
    │   ├── trainer_sft.py
    │   └── trainer_dpo.py
    ├── modelweights/        # Trained model outputs
    ├── results/             # Evaluation results
    ├── logs/                # Pipeline logs
    └── pipeline.py          # Main pipeline
"""

from .pipeline import Pipeline

__version__ = "1.0.0"
__all__ = ["Pipeline"]
