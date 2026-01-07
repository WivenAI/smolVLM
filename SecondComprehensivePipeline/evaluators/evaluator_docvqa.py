"""
DocVQA Evaluator - Evaluates document visual question answering
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .text_metrics import text_matches_any, calculate_anls
from .dpo_utils import BenchmarkDatasetIterator, ensure_model_loaded, extract_question, extract_all_answers

logger = logging.getLogger(__name__)


class DocVQAEvaluator(BaseEvaluator):
    """Evaluator for DocVQA dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "nielsr/docvqa_1200_examples"

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on DocVQA dataset"""
        ensure_model_loaded(self, model_path)

        logger.info("Evaluating on DocVQA...")
        dataset = self.load_cached_dataset(self.dataset_name, "train", max_samples)

        results = []
        iterator = BenchmarkDatasetIterator(dataset, self.load_image, "DocVQA")

        for item, image in tqdm(iterator, desc="DocVQA", total=len(dataset)):
            # Use dataloader extraction methods (will raise KeyError if field not found)
            question = extract_question(item, "query")
            answers = extract_all_answers(item, "answers")

            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": answers,
                "dataset": "docvqa"
            })

        iterator.log_skip_summary()
        accuracy, anls = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "docvqa", )

        return {
            "benchmark": "docvqa",
            "accuracy": accuracy,
            "anls": anls,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> tuple:
        """
        Calculate DocVQA accuracy and ANLS.

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
                response = str(result['response'])
                gt = result['ground_truth']
                gt_list = gt if isinstance(gt, list) else [gt]

                # Unidirectional: only check if gt is IN response
                matched = text_matches_any(response, gt_list)

                if matched:
                    correct += 1
                total += 1

                # Collect for ANLS calculation
                predictions.append(response)
                ground_truths.append(gt_list)

        accuracy = (correct / total * 100) if total > 0 else 0.0

        # Calculate ANLS
        avg_anls, _ = calculate_anls(predictions, ground_truths)

        return accuracy, avg_anls
