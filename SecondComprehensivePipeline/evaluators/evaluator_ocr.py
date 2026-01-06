"""
OCRBench Evaluator - Evaluates OCR capabilities
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .qcm_accuracy import normalize_text
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
        accuracy = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "ocrbench", )

        return {
            "benchmark": "ocrbench",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
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
