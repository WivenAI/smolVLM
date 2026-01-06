"""
Shared utilities for DPO dataset evaluation.

This module re-exports common utilities from the dataloader module
and provides evaluator-specific helpers.
"""

import logging
from typing import Optional

# Re-export from dataloader module (single source of truth)
from dataloader.dpo_dataset import (
    DPODatasetIterator,
    load_dpo_dataset,
)
from dataloader.benchmark_dataset import (
    BenchmarkDatasetIterator,
    BenchmarkMixin,
    BENCHMARK_CONFIGS,
)
from dataloader.base_dataset import ImageUtils

logger = logging.getLogger(__name__)

# Default constants for evaluation
DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_SUBSET_SEED = 42


def load_and_resize_image(image_path, max_size: int = DEFAULT_MAX_IMAGE_SIZE):
    """
    Load an image and resize if necessary.
    Delegates to ImageUtils from dataloader module.
    """
    image = ImageUtils.load_image(image_path)
    if image is None:
        return None
    return ImageUtils.resize_image(image, max_size)


def ensure_model_loaded(evaluator, model_path: Optional[str] = None) -> None:
    """
    Ensure the evaluator has a model loaded.

    Args:
        evaluator: BaseEvaluator instance
        model_path: Optional path to model weights
    """
    if model_path:
        evaluator.load_model(model_path)
    elif evaluator.model is None:
        evaluator.load_base_model()


# For backwards compatibility, expose extraction methods
extract_question = BenchmarkMixin.extract_question
extract_answer = BenchmarkMixin.extract_answer
extract_all_answers = BenchmarkMixin.extract_all_answers
