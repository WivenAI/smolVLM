"""
Shared QCM Accuracy Module

This module provides a single, unified function for calculating QCM accuracy
that is used by all evaluators and trainers for consistency.

Usage:
    from evaluators.qcm_accuracy import calculate_qcm_accuracy, extract_answer_letter

    # Extract answer letter from response
    letter = extract_answer_letter(response, ["A", "B", "C", "D"])

    # Calculate accuracy on results
    accuracy = calculate_qcm_accuracy(results, split="test")
"""

import re
import logging
from typing import List, Dict, Optional

from .answer_evaluator import AnswerExtractor

logger = logging.getLogger(__name__)

# Singleton extractor instance for performance
_extractor: Optional[AnswerExtractor] = None


def get_extractor() -> AnswerExtractor:
    """Get or create the singleton AnswerExtractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = AnswerExtractor(use_eleutherai=True)
    return _extractor


def extract_answer_letter(response: str, valid_options: List[str], question: Optional[str] = None) -> str:
    """
    Extract the answer letter from an LLM response.

    Uses the shared AnswerExtractor with all French and English patterns.

    Args:
        response: The LLM's response text
        valid_options: List of valid option letters (e.g., ["A", "B", "C", "D"])
        question: Optional question text (used for context-aware extraction)

    Returns:
        The extracted answer letter in uppercase, or empty string if not found
    """
    extractor = get_extractor()
    return extractor.extract(response, valid_options, question)


def normalize_text(text: str) -> str:
    """
    Normalize text for lenient comparison.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, no punctuation, no whitespace)
    """
    text = str(text).lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove all whitespace
    text = re.sub(r'\s+', '', text)
    return text


def calculate_qcm_accuracy(
    results: List[Dict],
    split: str = "full",
    log_to_wandb: bool = False,
    wandb_prefix: str = "eval"
) -> Dict[str, float]:
    """
    Calculate QCM accuracy with lenient matching.

    This is the single source of truth for QCM accuracy calculation.
    All evaluators and trainers should use this function.

    Args:
        results: List of result dictionaries, each containing:
            - response: The model's response
            - ground_truth OR correct_answer: The correct answer
            - predicted_letter: (optional) Pre-extracted predicted letter
            - options: (optional) Dict of option letters to option text
        split: One of "train", "test", or "full" - used for logging
        log_to_wandb: Whether to log metrics to wandb
        wandb_prefix: Prefix for wandb metric names

    Returns:
        Dictionary with accuracy metrics:
            - accuracy: Overall accuracy percentage
            - correct: Number of correct answers
            - total: Total number of samples
            - split: The split name (train/test/full)
    """
    if not results:
        return {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "split": split
        }

    correct = 0
    total = len(results)

    for r in results:
        # Get the response and ground truth
        response = r.get('response', '')
        ground_truth = r.get('ground_truth', r.get('correct_answer', ''))

        # Get or extract the predicted letter
        if 'predicted_letter' in r and r['predicted_letter']:
            predicted_letter = r['predicted_letter'].upper()
        else:
            # Extract from response using valid options
            options = r.get('options', {})
            valid_options = list(options.keys()) if options else ['A', 'B', 'C', 'D']
            predicted_letter = extract_answer_letter(response, valid_options)
            r['predicted_letter'] = predicted_letter  # Store for later use

        # Get correct letter
        correct_letter = ground_truth.upper()[0] if ground_truth else ''

        # Check 1: Exact letter match (strict)
        if predicted_letter and predicted_letter == correct_letter:
            r['is_correct'] = True
            correct += 1
            continue

        # Check 2: Lenient text matching (if options available)
        options = r.get('options', {})
        if options and correct_letter in options:
            correct_text = normalize_text(options[correct_letter])
            response_norm = normalize_text(response)

            # Bidirectional matching: correct in response OR response in correct
            if correct_text and response_norm:
                if correct_text in response_norm or response_norm in correct_text:
                    r['is_correct'] = True
                    correct += 1
                    continue

        # Check 3: Direct text comparison (without options)
        if not options and ground_truth:
            response_norm = normalize_text(response)
            gt_norm = normalize_text(ground_truth)

            if response_norm and gt_norm:
                if gt_norm in response_norm or response_norm in gt_norm:
                    r['is_correct'] = True
                    correct += 1
                    continue

        r['is_correct'] = False

    accuracy = (correct / total * 100) if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "split": split
    }

    # Log to wandb if requested
    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {
                    f"{wandb_prefix}/{split}_accuracy": accuracy,
                    f"{wandb_prefix}/{split}_correct": correct,
                    f"{wandb_prefix}/{split}_total": total,
                }
                wandb.log(wandb_metrics)
                logger.info(f"Logged {split} accuracy to wandb: {accuracy:.2f}%")
        except ImportError:
            pass

    logger.info(f"[{split.upper()}] Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return metrics


def calculate_accuracy_train_test(
    train_results: List[Dict],
    test_results: List[Dict],
    log_to_wandb: bool = False,
    wandb_prefix: str = "eval",
    global_step: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate accuracy for both train and test sets and detect overfitting.

    Args:
        train_results: Results from training set evaluation
        test_results: Results from test set evaluation
        log_to_wandb: Whether to log metrics to wandb
        wandb_prefix: Prefix for wandb metric names
        global_step: Current training step for wandb logging

    Returns:
        Dictionary with train, test, and combined metrics:
            - train: Train set metrics
            - test: Test set metrics
            - full: Combined metrics
            - train_test_gap: Difference between train and test accuracy
    """
    train_metrics = calculate_qcm_accuracy(train_results, split="train")
    test_metrics = calculate_qcm_accuracy(test_results, split="test")

    # Calculate full (combined) metrics
    all_results = train_results + test_results
    full_metrics = calculate_qcm_accuracy(all_results, split="full")

    # Calculate train-test gap
    train_test_gap = train_metrics["accuracy"] - test_metrics["accuracy"]

    # Detect overfitting/underfitting
    if train_test_gap > 10:
        logger.warning(f"Large train-test gap ({train_test_gap:.2f}%): possible OVERFITTING")
    elif train_metrics["accuracy"] > 90 and test_metrics["accuracy"] > 80:
        logger.info(f"Good generalization (train: {train_metrics['accuracy']:.1f}%, test: {test_metrics['accuracy']:.1f}%)")
    elif train_metrics["accuracy"] < 50:
        logger.warning(f"Low train accuracy ({train_metrics['accuracy']:.1f}%): possible UNDERFITTING")

    result = {
        "train": train_metrics,
        "test": test_metrics,
        "full": full_metrics,
        "train_test_gap": train_test_gap
    }

    # Log to wandb if requested
    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {
                    f"{wandb_prefix}/train_accuracy": train_metrics["accuracy"],
                    f"{wandb_prefix}/test_accuracy": test_metrics["accuracy"],
                    f"{wandb_prefix}/full_accuracy": full_metrics["accuracy"],
                    f"{wandb_prefix}/train_test_gap": train_test_gap,
                }
                if global_step is not None:
                    wandb.log(wandb_metrics, step=global_step)
                else:
                    wandb.log(wandb_metrics)
                logger.info(f"Logged train/test/full accuracy to wandb")
        except ImportError:
            pass

    return result
