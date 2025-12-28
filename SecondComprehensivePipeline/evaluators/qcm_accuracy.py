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
    Calculate QCM accuracy with BOTH strict and lenient matching.

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
            - accuracy: Strict letter matching accuracy (main metric)
            - strict_accuracy: Same as accuracy (for clarity)
            - lenient_accuracy: Letter match OR text-based match
            - correct: Number of correct answers (strict)
            - strict_correct: Same as correct
            - lenient_correct: Number correct with lenient matching
            - total: Total number of samples
            - split: The split name (train/test/full)
    """
    if not results:
        return {
            "accuracy": 0.0,
            "strict_accuracy": 0.0,
            "lenient_accuracy": 0.0,
            "correct": 0,
            "strict_correct": 0,
            "lenient_correct": 0,
            "total": 0,
            "extraction_failures": 0,
            "split": split
        }

    strict_correct = 0
    lenient_correct = 0
    total = len(results)
    extraction_failures = 0

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

        # Track extraction failures
        if not predicted_letter:
            extraction_failures += 1

        # Get correct letter
        correct_letter = ground_truth.upper()[0] if ground_truth else ''

        # STRICT: Exact letter match only
        is_strict_correct = predicted_letter and predicted_letter == correct_letter
        if is_strict_correct:
            strict_correct += 1
            lenient_correct += 1  # Strict match also counts for lenient
            r['is_correct'] = True
            r['is_correct_strict'] = True
            r['is_correct_lenient'] = True
            continue

        # LENIENT: Text-based matching (if strict failed)
        is_lenient_correct = False
        options = r.get('options', {})

        # Check 1: Lenient text matching (if options available)
        if options and correct_letter in options:
            correct_text = normalize_text(options[correct_letter])
            response_norm = normalize_text(response)

            # Bidirectional matching: correct in response OR response in correct
            if correct_text and response_norm:
                if correct_text in response_norm or response_norm in correct_text:
                    is_lenient_correct = True

        # Check 2: Direct text comparison (without options)
        if not is_lenient_correct and not options and ground_truth:
            response_norm = normalize_text(response)
            gt_norm = normalize_text(ground_truth)

            if response_norm and gt_norm:
                if gt_norm in response_norm or response_norm in gt_norm:
                    is_lenient_correct = True

        if is_lenient_correct:
            lenient_correct += 1
            r['is_correct'] = False  # Strict is false
            r['is_correct_strict'] = False
            r['is_correct_lenient'] = True
        else:
            r['is_correct'] = False
            r['is_correct_strict'] = False
            r['is_correct_lenient'] = False

    strict_accuracy = (strict_correct / total * 100) if total > 0 else 0.0
    lenient_accuracy = (lenient_correct / total * 100) if total > 0 else 0.0

    metrics = {
        "accuracy": strict_accuracy,  # Main metric is strict
        "strict_accuracy": strict_accuracy,
        "lenient_accuracy": lenient_accuracy,
        "correct": strict_correct,
        "strict_correct": strict_correct,
        "lenient_correct": lenient_correct,
        "total": total,
        "extraction_failures": extraction_failures,
        "split": split
    }

    # Log to wandb if requested
    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb_metrics = {
                    f"{wandb_prefix}/{split}_accuracy": strict_accuracy,
                    f"{wandb_prefix}/{split}_strict_accuracy": strict_accuracy,
                    f"{wandb_prefix}/{split}_lenient_accuracy": lenient_accuracy,
                    f"{wandb_prefix}/{split}_correct": strict_correct,
                    f"{wandb_prefix}/{split}_strict_correct": strict_correct,
                    f"{wandb_prefix}/{split}_lenient_correct": lenient_correct,
                    f"{wandb_prefix}/{split}_total": total,
                    f"{wandb_prefix}/{split}_extraction_failures": extraction_failures,
                }
                wandb.log(wandb_metrics)
                logger.info(f"Logged {split} accuracies to wandb - Strict: {strict_accuracy:.2f}%, Lenient: {lenient_accuracy:.2f}%")
        except ImportError:
            pass

    log_msg = f"[{split.upper()}] Strict: {strict_accuracy:.2f}% ({strict_correct}/{total}), Lenient: {lenient_accuracy:.2f}% ({lenient_correct}/{total})"
    if extraction_failures > 0:
        log_msg += f" [extraction_failures: {extraction_failures}]"
    logger.info(log_msg)

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
                    f"{wandb_prefix}/train_extraction_failures": train_metrics["extraction_failures"],
                    f"{wandb_prefix}/test_extraction_failures": test_metrics["extraction_failures"],
                    f"{wandb_prefix}/full_extraction_failures": full_metrics["extraction_failures"],
                }
                if global_step is not None:
                    wandb.log(wandb_metrics, step=global_step)
                else:
                    wandb.log(wandb_metrics)
                logger.info(f"Logged train/test/full accuracy to wandb")
        except ImportError:
            pass

    return result
