"""
ChartQA Evaluator - Evaluates chart understanding and question answering
"""

import re
from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .qcm_accuracy import normalize_text
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
        accuracy = self.calculate_accuracy(results)

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "chartqa", )

        return {
            "benchmark": "chartqa",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": iterator.get_total_skipped(),
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

                # Text comparison with bidirectional matching
                response_norm = normalize_text(response)
                gt_norm = normalize_text(gt)

                # Bidirectional: gt in response OR response in gt OR exact match
                if gt_norm and response_norm:
                    if gt_norm in response_norm or response_norm in gt_norm or gt_norm == response_norm:
                        correct += 1

                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def _extract_number(self, text: str) -> float:
        """Extract first number from text"""
        # Remove commas and percentage signs
        text = text.replace(',', '').replace('%', '')
        # Find numbers
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        return None
