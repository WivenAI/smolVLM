"""
OCRBench Evaluator - Evaluates OCR capabilities
"""

import re
from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison:
    - Convert to lowercase
    - Remove spaces, tabs, newlines
    """
    text = str(text).lower()
    # Remove all whitespace (spaces, tabs, newlines)
    text = re.sub(r'\s+', '', text)
    return text


class OCRBenchEvaluator(BaseEvaluator):
    """Evaluator for OCRBench dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "echo840/OCRBench"

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on OCRBench dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        logger.info("Evaluating on OCRBench...")
        dataset = self.load_cached_dataset(self.dataset_name, "test", max_samples)

        results = []
        skipped_no_path = 0
        skipped_load_failed = 0
        for item in tqdm(dataset, desc="OCRBench"):
            image_path = item.get('image_path')
            if not image_path:
                skipped_no_path += 1
                continue

            image = self.load_image(image_path)
            if image is None:
                skipped_load_failed += 1
                continue

            question = item['question']
            ground_truth = item.get('answer', '')
            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "task_type": item.get('question_type', item.get('task_type', 'ocr')),
                "dataset": item.get('dataset', 'ocrbench')
            })

        # Log skipped samples summary
        total_skipped = skipped_no_path + skipped_load_failed
        if total_skipped > 0:
            logger.warning(f"OCRBench: Skipped {total_skipped} samples ({skipped_no_path} no path, {skipped_load_failed} load failed)")

        accuracy = self.calculate_accuracy(results)

        return {
            "benchmark": "ocrbench",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": total_skipped,
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate OCRBench accuracy with lenient bidirectional matching.
        Checks if ground truth is in prediction OR prediction is in ground truth.
        Uses normalized text (lowercase, no whitespace) for easier matching.
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = normalize_text(result['response'])
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                matched = False
                for gt in ground_truths:
                    gt_normalized = normalize_text(gt)
                    # Bidirectional check: gt in response OR response in gt
                    if gt_normalized and response:
                        if gt_normalized in response or response in gt_normalized:
                            matched = True
                            break
                        # Also check exact match
                        if gt_normalized == response:
                            matched = True
                            break

                if matched:
                    correct += 1
                total += 1

        return (correct / total * 100) if total > 0 else 0.0
