"""
Shared utilities for DPO dataset evaluation.

This module provides common functions used by LogProb, ROUGE, and BERTScore evaluators
to avoid code duplication.
"""

import json
import random
import logging
from typing import Dict, Any, List, Optional, Iterator, Tuple
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

# Default constants for evaluation
DEFAULT_MAX_IMAGE_SIZE = 1024
DEFAULT_SUBSET_SEED = 42


def load_dpo_dataset(
    dataset_path: str,
    max_samples: Optional[int] = None,
    use_fixed_subset: bool = False,
    subset_seed: int = DEFAULT_SUBSET_SEED
) -> List[Dict]:
    """
    Load DPO dataset from JSON file with optional fixed subset selection.

    Args:
        dataset_path: Path to the DPO dataset JSON file
        max_samples: Maximum number of samples to return (None for all)
        use_fixed_subset: If True, use fixed random subset for reproducibility
        subset_seed: Seed for reproducible subset selection

    Returns:
        List of dataset items (dictionaries with 'image_name', 'prompt', 'chosen', 'rejected')
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_samples and max_samples < len(dataset):
        if use_fixed_subset:
            # Use fixed seed for consistent subset across all evaluations
            rng = random.Random(subset_seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            selected_indices = sorted(indices[:max_samples])
            dataset = [dataset[i] for i in selected_indices]
            logger.info(f"Using fixed subset of {len(dataset)} samples (seed={subset_seed})")
        else:
            dataset = dataset[:max_samples]

    return dataset


def load_and_resize_image(
    image_path: Path,
    max_size: int = DEFAULT_MAX_IMAGE_SIZE
) -> Optional[Image.Image]:
    """
    Load an image and resize if necessary.

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) for the image

    Returns:
        PIL Image in RGB mode, or None if image doesn't exist
    """
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert('RGB')

    # Resize large images to prevent memory issues
    if image.size[0] > max_size or image.size[1] > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return image


class DPODatasetIterator:
    """
    Iterator for DPO datasets that handles image loading and skip counting.

    Usage:
        iterator = DPODatasetIterator(dataset, image_dir, "ROUGE")
        for item, image in iterator:
            # Process item and image

        # Get skip statistics
        skipped_images, skipped_errors = iterator.get_skip_counts()
        iterator.log_skip_summary()
    """

    def __init__(
        self,
        dataset: List[Dict],
        image_dir: Path,
        evaluator_name: str = "Evaluator",
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE
    ):
        """
        Initialize the iterator.

        Args:
            dataset: List of DPO dataset items
            image_dir: Directory containing images
            evaluator_name: Name for logging (e.g., "ROUGE", "LogProb")
            max_image_size: Maximum image dimension
        """
        self.dataset = dataset
        self.image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir
        self.evaluator_name = evaluator_name
        self.max_image_size = max_image_size
        self.skipped_missing_image = 0
        self.skipped_error = 0

    def __iter__(self) -> Iterator[Tuple[Dict, Image.Image]]:
        """Iterate over dataset, yielding (item, image) tuples."""
        for item in self.dataset:
            try:
                image_path = self.image_dir / item['image_name']
                image = load_and_resize_image(image_path, self.max_image_size)

                if image is None:
                    logger.debug(f"Image not found: {image_path}")
                    self.skipped_missing_image += 1
                    continue

                yield item, image

            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                self.skipped_error += 1
                continue

    def get_skip_counts(self) -> Tuple[int, int]:
        """
        Get the counts of skipped samples.

        Returns:
            Tuple of (skipped_missing_image, skipped_error)
        """
        return self.skipped_missing_image, self.skipped_error

    def log_skip_summary(self) -> None:
        """Log a summary of skipped samples if any were skipped."""
        total_skipped = self.skipped_missing_image + self.skipped_error
        if total_skipped > 0:
            logger.warning(
                f"{self.evaluator_name}: Skipped {total_skipped} samples "
                f"({self.skipped_missing_image} missing images, {self.skipped_error} errors)"
            )


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
