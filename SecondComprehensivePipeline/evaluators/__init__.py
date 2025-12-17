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
]
