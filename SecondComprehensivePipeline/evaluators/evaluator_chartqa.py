"""
ChartQA Evaluator - Evaluates chart understanding and question answering
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .text_metrics import text_contains_answer, compare_numeric, normalize_text, calculate_anls
from .dpo_utils import BenchmarkDatasetIterator, ensure_model_loaded, extract_question, extract_answer

logger = logging.getLogger(__name__)


class ChartQAEvaluator(BaseEvaluator):
    """Evaluator for ChartQA dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "HuggingFaceM4/ChartQA"

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on ChartQA dataset"""
        ensure_model_loaded(self, model_path)

        logger.info("Evaluating on ChartQA...")
        dataset = self.load_cached_dataset(self.dataset_name, "test", max_samples)

        results = []
        iterator = BenchmarkDatasetIterator(dataset, self.load_image, "ChartQA")

        for item, image in tqdm(iterator, desc="ChartQA", total=len(dataset)):
            # Use dataloader extraction methods (will raise KeyError if field not found)
            question = extract_question(item, "query")
            answer = extract_answer(item, "label")

            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": answer,
                "dataset": "chartqa"
            })

        iterator.log_skip_summary()
        accuracy, anls = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "chartqa", )

        return {
            "benchmark": "chartqa",
            "accuracy": accuracy,
            "anls": anls,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> tuple:
        """
        Calculate ChartQA accuracy and ANLS.

        Handles numeric values (5% tolerance) and text answers.
        Uses UNIDIRECTIONAL matching: only checks if ground truth is IN prediction.
        Does NOT check if prediction is in ground truth (no bidirectional matching).

        Returns:
            Tuple of (accuracy percentage, ANLS score 0-1)
        """
        if not results:
            return 0.0, 0.0

        correct = 0
        total = 0
        predictions = []
        ground_truths = []

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                gt = str(result['ground_truth']).lower().strip()

                # Collect for ANLS calculation
                predictions.append(response)
                ground_truths.append(gt)

                # Try numeric comparison first (5% tolerance)
                numeric_match = compare_numeric(response, gt, tolerance=0.05)
                if numeric_match is True:
                    correct += 1
                    total += 1
                    continue

                # Text comparison with UNIDIRECTIONAL matching only
                # Only check: gt in response (NOT response in gt)
                if text_contains_answer(response, gt):
                    correct += 1

                total += 1

        accuracy = (correct / total * 100) if total > 0 else 0.0

        # Calculate ANLS
        avg_anls, _ = calculate_anls(predictions, ground_truths)

        return accuracy, avg_anls
