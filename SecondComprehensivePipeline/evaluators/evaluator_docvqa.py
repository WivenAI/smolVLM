"""
DocVQA Evaluator - Evaluates document visual question answering
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging
import re

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class DocVQAEvaluator(BaseEvaluator):
    """Evaluator for DocVQA dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "nielsr/docvqa_1200_examples"  # Fixed: original requires auth

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on DocVQA dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        logger.info("Evaluating on DocVQA...")
        dataset = self.load_cached_dataset(self.dataset_name, "train", max_samples)

        results = []
        for item in tqdm(dataset, desc="DocVQA"):
            image_path = item.get('image_path')
            if not image_path:
                continue

            image = self.load_image(image_path)
            if image is None:
                continue

            question = item.get('question', item.get('query', ''))
            answers = item.get('answers', item.get('answer', []))
            if isinstance(answers, str):
                answers = [answers]

            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": answers,
                "dataset": "docvqa"
            })

        accuracy = self.calculate_accuracy(results)

        return {
            "benchmark": "docvqa",
            "accuracy": accuracy,
            "total_samples": len(results),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate DocVQA accuracy using ANLS (Average Normalized Levenshtein Similarity)
        Simplified version: exact match or contains check
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = self._normalize_text(str(result['response']))
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                # Check if any ground truth matches
                matched = False
                for gt in ground_truths:
                    gt_norm = self._normalize_text(str(gt))
                    if gt_norm in response or response in gt_norm:
                        matched = True
                        break
                    # Also check exact match
                    if gt_norm == response:
                        matched = True
                        break

                if matched:
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
