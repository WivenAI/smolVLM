"""
Shared Text Metrics Module

Provides unified text comparison metrics for all evaluators:
- ANLS (Average Normalized Levenshtein Similarity)
- Unidirectional containment matching (gt in response only)
- Numeric comparison with tolerance

Usage:
    from evaluators.text_metrics import calculate_anls, text_contains_answer, compare_numeric
"""

import re
from typing import List, Union, Optional, Tuple


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

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


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def calculate_nls(prediction: str, ground_truth: str, threshold: float = 0.5) -> float:
    """
    Calculate Normalized Levenshtein Similarity (NLS) between prediction and ground truth.

    NLS = max(1 - edit_distance / max_len, 0) if NLS >= threshold else 0

    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer
        threshold: Minimum similarity threshold (default 0.5 as per DocVQA standard)

    Returns:
        NLS score between 0 and 1
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)

    if not pred_norm and not gt_norm:
        return 1.0
    if not pred_norm or not gt_norm:
        return 0.0

    max_len = max(len(pred_norm), len(gt_norm))
    edit_dist = levenshtein_distance(pred_norm, gt_norm)

    nls = 1.0 - (edit_dist / max_len)

    # Apply threshold
    return nls if nls >= threshold else 0.0


def calculate_anls(
    predictions: List[str],
    ground_truths: List[Union[str, List[str]]],
    threshold: float = 0.5
) -> Tuple[float, List[float]]:
    """
    Calculate Average Normalized Levenshtein Similarity (ANLS) for a batch.

    For each sample, if ground_truth is a list, takes the max NLS across all ground truths.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truths (each can be str or list of str)
        threshold: Minimum similarity threshold (default 0.5)

    Returns:
        Tuple of (average ANLS, list of per-sample NLS scores)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")

    if not predictions:
        return 0.0, []

    nls_scores = []

    for pred, gt in zip(predictions, ground_truths):
        # Handle multiple ground truths (take max)
        if isinstance(gt, list):
            nls = max(calculate_nls(pred, g, threshold) for g in gt) if gt else 0.0
        else:
            nls = calculate_nls(pred, gt, threshold)
        nls_scores.append(nls)

    avg_anls = sum(nls_scores) / len(nls_scores)
    return avg_anls, nls_scores


def text_contains_answer(prediction: str, ground_truth: str) -> bool:
    """
    Check if prediction contains the ground truth answer.

    UNIDIRECTIONAL: Only checks if ground truth is IN the prediction.
    Does NOT check if prediction is in ground truth (removes bidirectional matching).

    Args:
        prediction: Model's predicted response
        ground_truth: Expected correct answer

    Returns:
        True if normalized ground truth is contained in normalized prediction
    """
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)

    if not pred_norm or not gt_norm:
        return False

    # Only check: ground truth IN prediction (NOT prediction in ground truth)
    return gt_norm in pred_norm


def text_matches_any(prediction: str, ground_truths: Union[str, List[str]]) -> bool:
    """
    Check if prediction matches any of the ground truths.

    Uses unidirectional containment: gt in prediction only.

    Args:
        prediction: Model's predicted response
        ground_truths: Single ground truth or list of ground truths

    Returns:
        True if any ground truth is contained in prediction
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    pred_norm = normalize_text(prediction)
    if not pred_norm:
        return False

    for gt in ground_truths:
        gt_norm = normalize_text(gt)
        if gt_norm and gt_norm in pred_norm:
            return True
        # Also check exact match
        if gt_norm and gt_norm == pred_norm:
            return True

    return False


def extract_number(text: str) -> Optional[float]:
    """
    Extract the first number from text.

    Args:
        text: Text containing a number

    Returns:
        Extracted number as float, or None if no number found
    """
    # Remove commas and percentage signs
    text = text.replace(',', '').replace('%', '')
    # Find numbers (including negative and decimal)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


def compare_numeric(prediction: str, ground_truth: str, tolerance: float = 0.05) -> Optional[bool]:
    """
    Compare two values numerically with tolerance.

    Args:
        prediction: Model's predicted answer
        ground_truth: Expected correct answer
        tolerance: Relative tolerance (default 5%)

    Returns:
        True if numbers match within tolerance, False if they don't match,
        None if either value is not a valid number
    """
    pred_num = extract_number(prediction)
    gt_num = extract_number(ground_truth)

    if pred_num is None or gt_num is None:
        return None

    # Handle zero ground truth specially
    if gt_num == 0:
        return abs(pred_num) <= 0.01

    # Check relative tolerance
    return abs(pred_num - gt_num) <= abs(gt_num) * tolerance + 0.01
