"""
ROUGE Evaluator for DPO Dataset
Evaluates model responses against ground truth using ROUGE metrics
"""

import json
import random
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import torch
from PIL import Image

from .base_evaluator import BaseEvaluator

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
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        logger.info("Evaluating with ROUGE...")

        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Select subset using fixed seed for reproducibility
        if max_samples and max_samples < len(dataset):
            if use_fixed_subset:
                # Use fixed seed for consistent subset across all evaluations
                rng = random.Random(subset_seed)
                indices = list(range(len(dataset)))
                rng.shuffle(indices)
                selected_indices = sorted(indices[:max_samples])
                dataset = [dataset[i] for i in selected_indices]
                logger.info(f"Using fixed subset of {len(dataset)} samples (seed={subset_seed})")
            else:
                dataset = dataset[:max_samples]

        logger.info(f"Loaded {len(dataset)} DPO examples for ROUGE evaluation")

        results = []
        all_rouge1 = []
        all_rouge2 = []
        all_rougeL = []

        image_dir = Path(image_dir)
        skipped_missing_image = 0
        skipped_error = 0

        scorer = self._get_rouge_scorer()

        for item in tqdm(dataset, desc="ROUGE"):
            try:
                # Load image
                image_path = image_dir / item['image_name']
                if not image_path.exists():
                    logger.debug(f"Image not found: {image_path}")
                    skipped_missing_image += 1
                    continue

                image = Image.open(image_path).convert('RGB')

                # Resize large images
                max_size = 1024
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

                prompt = item['prompt']
                reference = item['chosen']  # Ground truth is the chosen response

                # Generate prediction
                prediction = self.generate_response(image, prompt)

                # Calculate ROUGE scores
                scores = scorer.score(reference, prediction)

                rouge1_f = scores['rouge1'].fmeasure
                rouge2_f = scores['rouge2'].fmeasure
                rougeL_f = scores['rougeL'].fmeasure

                all_rouge1.append(rouge1_f)
                all_rouge2.append(rouge2_f)
                all_rougeL.append(rougeL_f)

                results.append({
                    "image_name": item['image_name'],
                    "prompt": prompt,
                    "prediction": prediction,
                    "reference": reference,
                    "rouge1": rouge1_f,
                    "rouge2": rouge2_f,
                    "rougeL": rougeL_f
                })

            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                skipped_error += 1
                continue

        # Log skipped samples summary
        total_skipped = skipped_missing_image + skipped_error
        if total_skipped > 0:
            logger.warning(f"ROUGE: Skipped {total_skipped} samples ({skipped_missing_image} missing images, {skipped_error} errors)")

        if not results:
            return {
                "benchmark": "rouge",
                "accuracy": 0.0,
                "total_samples": 0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "results": []
            }

        # Calculate means
        rouge1_mean = sum(all_rouge1) / len(all_rouge1)
        rouge2_mean = sum(all_rouge2) / len(all_rouge2)
        rougeL_mean = sum(all_rougeL) / len(all_rougeL)

        # Use ROUGE-L as the "accuracy" metric for comparison (scaled to percentage)
        accuracy = rougeL_mean * 100

        return {
            "benchmark": "rouge",
            "accuracy": accuracy,
            "total_samples": len(results),
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
