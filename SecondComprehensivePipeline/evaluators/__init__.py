"""
Evaluation module for NoHallucinations pipeline
"""

from .base_evaluator import BaseEvaluator
from .evaluator_ocr import OCRBenchEvaluator
from .evaluator_docvqa import DocVQAEvaluator
from .evaluator_chartqa import ChartQAEvaluator
from .evaluator_qcm import QCMEvaluator
from .evaluator_logprob import LogProbEvaluator
from .evaluator_bertscore import BertScoreEvaluator
from .evaluator_rouge import RougeEvaluator
from .evaluator_all import EvaluatorAll, evaluate_model
from .qcm_accuracy import (
    extract_answer_letter,
    calculate_qcm_accuracy,
    calculate_accuracy_train_test,
    normalize_text,
)
from .answer_evaluator import AnswerExtractor, extract_answer
from .dpo_utils import (
    load_dpo_dataset,
    load_and_resize_image,
    DPODatasetIterator,
    BenchmarkDatasetIterator,
    ensure_model_loaded,
)

__all__ = [
    "BaseEvaluator",
    "OCRBenchEvaluator",
    "DocVQAEvaluator",
    "ChartQAEvaluator",
    "QCMEvaluator",
    "LogProbEvaluator",
    "BertScoreEvaluator",
    "RougeEvaluator",
    "EvaluatorAll",
    "evaluate_model",
    # Shared QCM accuracy utilities
    "extract_answer_letter",
    "calculate_qcm_accuracy",
    "calculate_accuracy_train_test",
    "normalize_text",
    "AnswerExtractor",
    "extract_answer",
    # Shared dataset utilities
    "load_dpo_dataset",
    "load_and_resize_image",
    "DPODatasetIterator",
    "BenchmarkDatasetIterator",
    "ensure_model_loaded",
]
