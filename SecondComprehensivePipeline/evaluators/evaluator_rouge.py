"""
ROUGE Evaluator for DPO Dataset
Evaluates model responses against ground truth using ROUGE metrics
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging

from .base_evaluator import BaseEvaluator
from .dpo_utils import load_dpo_dataset, DPODatasetIterator, ensure_model_loaded

logger = logging.getLogger(__name__)


class RougeEvaluator(BaseEvaluator):
    """Evaluator using ROUGE to compare model responses with ground truth (chosen responses)"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self._rouge_scorer = None

    def _get_rouge_scorer(self):
        """Lazy load rouge_score to avoid import issues"""
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer
                # Compute all ROUGE variants: ROUGE-1, ROUGE-2, ROUGE-L
                self._rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
                )
            except ImportError:
                logger.error("rouge-score not installed. Run: pip install rouge-score")
                raise
        return self._rouge_scorer

    def evaluate(self, dataset_path: str = None, image_dir: str = None,
                 model_path: str = None, max_samples: int = None,
                 use_fixed_subset: bool = False, subset_seed: int = 42) -> Dict[str, Any]:
        """Evaluate using ROUGE on DPO dataset (comparing model output with chosen response)

        Args:
            dataset_path: Path to the DPO dataset JSON file
            image_dir: Directory containing images
            model_path: Path to model weights (optional)
            max_samples: Maximum number of samples to evaluate
            use_fixed_subset: If True, use a fixed random subset for consistent evaluation
            subset_seed: Seed for reproducible subset selection (default: 42)
        """
        ensure_model_loaded(self, model_path)
        logger.info("Evaluating with ROUGE...")

        # Load dataset using shared utility
        dataset = load_dpo_dataset(dataset_path, max_samples, use_fixed_subset, subset_seed)
        logger.info(f"Loaded {len(dataset)} DPO examples for ROUGE evaluation")

        results = []
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []

        scorer = self._get_rouge_scorer()

        # Iterate using shared iterator
        iterator = DPODatasetIterator(dataset, image_dir, "ROUGE")

        for item, image in tqdm(iterator, desc="ROUGE", total=len(dataset)):
            prompt = item['prompt']
            reference = item['chosen']  # Ground truth is the chosen response

            # Generate prediction
            prediction = self.generate_response(image, prompt)

            # Calculate all ROUGE scores
            scores = scorer.score(reference, prediction)
            rouge1_f = scores['rouge1'].fmeasure
            rouge2_f = scores['rouge2'].fmeasure
            rougeL_f = scores['rougeL'].fmeasure

            all_rouge1.append(rouge1_f)
            all_rouge2.append(rouge2_f)
            all_rougeL.append(rougeL_f)

            results.append({
                "image_name": item['image_name'],
                "question": prompt,  # Add for sample formatter
                "response": prediction,  # Add for sample formatter
                "ground_truth": reference,  # Add for sample formatter
                "prompt": prompt,
                "prediction": prediction,
                "reference": reference,
                "rouge1": rouge1_f,
                "rouge2": rouge2_f,
                "rougeL": rougeL_f
            })

        # Log skip summary
        iterator.log_skip_summary()

        # Get skipped counts from iterator
        skipped_missing, skipped_errors = iterator.get_skip_counts()
        skipped_samples = skipped_missing + skipped_errors

        if not results:
            return {
                "benchmark": "rouge",
                "accuracy": 0.0,
                "total_samples": 0,
                "skipped_samples": skipped_samples,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "results": []
            }

        # Calculate mean ROUGE scores
        rouge1_mean = sum(all_rouge1) / len(all_rouge1)
        rouge2_mean = sum(all_rouge2) / len(all_rouge2)
        rougeL_mean = sum(all_rougeL) / len(all_rougeL)

        # Use ROUGE-L as the "accuracy" metric for comparison (scaled to percentage)
        accuracy = rougeL_mean * 100

        # Save and print sample Q&A
        if results:
            from pathlib import Path
            output_dir = Path(self.cache_dir).parent / "evaluation_samples"
            self.save_and_print_samples(results, str(output_dir), "rouge", num_samples=3)

        return {
            "benchmark": "rouge",
            "accuracy": accuracy,
            "total_samples": len(results),
            "skipped_samples": skipped_samples,
            "rouge1": rouge1_mean,
            "rouge2": rouge2_mean,
            "rougeL": rougeL_mean,
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate average ROUGE-L score from results"""
        if not results:
            return 0.0
        rougeL_scores = [r.get('rougeL', 0.0) for r in results]
        return (sum(rougeL_scores) / len(rougeL_scores) * 100) if rougeL_scores else 0.0
