"""
BERTScore Evaluator for DPO Dataset
Evaluates model responses against ground truth using BERTScore metrics
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import torch
from PIL import Image

from .base_evaluator import BaseEvaluator

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
                 lang: str = "en") -> Dict[str, Any]:
        """Evaluate using BERTScore on DPO dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        logger.info("Evaluating with BERTScore...")

        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if max_samples:
            dataset = dataset[:max_samples]

        logger.info(f"Loaded {len(dataset)} DPO examples")

        results = []
        all_predictions = []
        all_references = []

        image_dir = Path(image_dir)

        for item in tqdm(dataset, desc="BERTScore"):
            try:
                # Load image
                image_path = image_dir / item['image_name']
                if not image_path.exists():
                    continue

                image = Image.open(image_path).convert('RGB')

                # Resize large images
                max_size = 1024
                if image.size[0] > max_size or image.size[1] > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

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

            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                continue

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
        import gc
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
