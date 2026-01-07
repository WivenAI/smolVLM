"""
OCRBench Evaluator - Evaluates OCR capabilities
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .text_metrics import text_matches_any, calculate_anls
from .dpo_utils import BenchmarkDatasetIterator, ensure_model_loaded, extract_question, extract_answer

logger = logging.getLogger(__name__)


class OCRBenchEvaluator(BaseEvaluator):
    """Evaluator for OCRBench dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "echo840/OCRBench"

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on OCRBench dataset"""
        ensure_model_loaded(self, model_path)

        logger.info("Evaluating on OCRBench...")
        dataset = self.load_cached_dataset(self.dataset_name, "test", max_samples)

        results = []
        iterator = BenchmarkDatasetIterator(dataset, self.load_image, "OCRBench")

        for item, image in tqdm(iterator, desc="OCRBench", total=len(dataset)):
            # Use dataloader extraction methods (will raise KeyError if field not found)
            question = extract_question(item, "question")
            ground_truth = extract_answer(item, "answer")
            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "task_type": item.get('question_type', item.get('task_type', 'ocr')),
                "dataset": item.get('dataset', 'ocrbench')
            })

        iterator.log_skip_summary()
        accuracy, anls = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "ocrbench", )

        return {
            "benchmark": "ocrbench",
            "accuracy": accuracy,
            "anls": anls,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> tuple:
        """
        Calculate OCRBench accuracy and ANLS.

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
                response = result['response']
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
