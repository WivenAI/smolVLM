"""
BERTScore Evaluator for DPO Dataset
Evaluates model responses against ground truth using BERTScore metrics
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging
import gc
import torch

from .base_evaluator import BaseEvaluator
from .dpo_utils import load_dpo_dataset, DPODatasetIterator, ensure_model_loaded

logger = logging.getLogger(__name__)


class BertScoreEvaluator(BaseEvaluator):
    """Evaluator using BERTScore to compare model responses with ground truth"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self._bert_scorer = None

    def _get_bert_score(self):
        """Lazy load bert_score to avoid import issues"""
        if self._bert_scorer is None:
            try:
                from bert_score import score as bert_score_fn
                self._bert_scorer = bert_score_fn
            except ImportError:
                logger.error("bert-score not installed. Run: pip install bert-score")
                raise
        return self._bert_scorer

    def evaluate(self, dataset_path: str = None, image_dir: str = None,
                 model_path: str = None, max_samples: int = None,
                 lang: str = "en", use_fixed_subset: bool = False,
                 subset_seed: int = 42) -> Dict[str, Any]:
        """Evaluate using BERTScore on DPO dataset

        Args:
            dataset_path: Path to the DPO dataset JSON file
            image_dir: Directory containing images
            model_path: Path to model weights (optional)
            max_samples: Maximum number of samples to evaluate
            lang: Language for BERTScore (default: "en")
            use_fixed_subset: If True, use a fixed random subset for consistent evaluation
            subset_seed: Seed for reproducible subset selection (default: 42)
        """
        ensure_model_loaded(self, model_path)
        logger.info("Evaluating with BERTScore...")

        # Load dataset using shared utility
        dataset = load_dpo_dataset(dataset_path, max_samples, use_fixed_subset, subset_seed)
        logger.info(f"Loaded {len(dataset)} DPO examples")

        results = []
        all_predictions = []
        all_references = []

        # Iterate using shared iterator
        iterator = DPODatasetIterator(dataset, image_dir, "BERTScore")

        for item, image in tqdm(iterator, desc="BERTScore", total=len(dataset)):
            prompt = item['prompt']
            reference = item['chosen']

            # Generate prediction
            prediction = self.generate_response(image, prompt)

            all_predictions.append(prediction)
            all_references.append(reference)

            results.append({
                "image_name": item['image_name'],
                "prompt": prompt,
                "prediction": prediction,
                "reference": reference
            })

        # Log skip summary
        iterator.log_skip_summary()

        if not all_predictions:
            return {
                "benchmark": "bertscore",
                "accuracy": 0.0,
                "total_samples": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "results": []
            }

        # Free GPU memory before running BERTScore
        logger.info("Freeing GPU memory before BERTScore computation...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate overall BERTScore on CPU to avoid OOM
        logger.info("Computing BERTScore metrics on CPU...")
        bert_score_fn = self._get_bert_score()

        P, R, F1 = bert_score_fn(
            all_predictions,
            all_references,
            lang=lang,
            verbose=True,
            device="cpu"  # Use CPU to avoid CUDA OOM
        )

        # Add scores to individual results
        for i, result in enumerate(results):
            result["bertscore"] = {
                "precision": float(P[i]),
                "recall": float(R[i]),
                "f1": float(F1[i])
            }

        # Calculate means
        precision_mean = float(P.mean())
        recall_mean = float(R.mean())
        f1_mean = float(F1.mean())

        # Use F1 as the "accuracy" metric for comparison
        accuracy = f1_mean * 100

        return {
            "benchmark": "bertscore",
            "accuracy": accuracy,
            "total_samples": len(results),
            "precision": precision_mean,
            "recall": recall_mean,
            "f1": f1_mean,
            "precision_std": float(P.std()),
            "recall_std": float(R.std()),
            "f1_std": float(F1.std()),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate average F1 score from results"""
        if not results:
            return 0.0
        f1_scores = [r.get('bertscore', {}).get('f1', 0.0) for r in results]
        return (sum(f1_scores) / len(f1_scores) * 100) if f1_scores else 0.0
