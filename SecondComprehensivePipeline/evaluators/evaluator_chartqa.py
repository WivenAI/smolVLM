"""
ChartQA Evaluator - Evaluates chart understanding and question answering
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging
import re

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class ChartQAEvaluator(BaseEvaluator):
    """Evaluator for ChartQA dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "HuggingFaceM4/ChartQA"

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on ChartQA dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        logger.info("Evaluating on ChartQA...")
        dataset = self.load_cached_dataset(self.dataset_name, "test", max_samples)

        results = []
        skipped_no_path = 0
        skipped_load_failed = 0
        for item in tqdm(dataset, desc="ChartQA"):
            image_path = item.get('image_path')
            if not image_path:
                skipped_no_path += 1
                continue

            image = self.load_image(image_path)
            if image is None:
                skipped_load_failed += 1
                continue

            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('label', ''))

            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": answer,
                "dataset": "chartqa"
            })

        # Log skipped samples summary
        total_skipped = skipped_no_path + skipped_load_failed
        if total_skipped > 0:
            logger.warning(f"ChartQA: Skipped {total_skipped} samples ({skipped_no_path} no path, {skipped_load_failed} load failed)")

        accuracy = self.calculate_accuracy(results)

        return {
            "benchmark": "chartqa",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": total_skipped,
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate ChartQA accuracy with relaxed matching
        Handles numeric values and text answers
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                gt = str(result['ground_truth']).lower().strip()

                # Try numeric comparison first
                try:
                    pred_num = self._extract_number(response)
                    gt_num = self._extract_number(gt)
                    if pred_num is not None and gt_num is not None:
                        # Allow 5% tolerance for numeric answers
                        if abs(pred_num - gt_num) <= abs(gt_num) * 0.05 + 0.01:
                            correct += 1
                            total += 1
                            continue
                except:
                    pass

                # Text comparison
                response_norm = self._normalize_text(response)
                gt_norm = self._normalize_text(gt)

                if gt_norm in response_norm or response_norm == gt_norm:
                    correct += 1

                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison:
        - Convert to lowercase
        - Remove punctuation
        - Remove all whitespace (spaces, tabs, newlines)
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # Remove all whitespace
        text = re.sub(r'\s+', '', text)
        return text

    def _extract_number(self, text: str) -> float:
        """Extract first number from text"""
        # Remove commas and percentage signs
        text = text.replace(',', '').replace('%', '')
        # Find numbers
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        return None
