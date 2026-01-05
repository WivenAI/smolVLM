"""
DocVQA Evaluator - Evaluates document visual question answering
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .qcm_accuracy import normalize_text
from .dpo_utils import BenchmarkDatasetIterator, ensure_model_loaded

logger = logging.getLogger(__name__)


class DocVQAEvaluator(BaseEvaluator):
    """Evaluator for DocVQA dataset"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_name = "nielsr/docvqa_1200_examples"  # Fixed: original requires auth

    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Evaluate on DocVQA dataset"""
        ensure_model_loaded(self, model_path)

        logger.info("Evaluating on DocVQA...")
        dataset = self.load_cached_dataset(self.dataset_name, "train", max_samples)

        results = []
        iterator = BenchmarkDatasetIterator(dataset, self.load_image, "DocVQA")

        for item, image in tqdm(iterator, desc="DocVQA", total=len(dataset)):
            # Handle query field which can be a dict with language keys or a string
            query_field = item.get('question', item.get('query', ''))
            if isinstance(query_field, dict):
                # Extract English text from multi-language query dict
                question = query_field.get('en', str(query_field))
            else:
                question = str(query_field)
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

        iterator.log_skip_summary()
        accuracy = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "docvqa", )

        return {
            "benchmark": "docvqa",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
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
                response = normalize_text(str(result['response']))
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                # Check if any ground truth matches
                matched = False
                for gt in ground_truths:
                    gt_norm = normalize_text(str(gt))
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
