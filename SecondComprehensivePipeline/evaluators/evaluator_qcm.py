"""
QCM Evaluator - Evaluates ERP multiple choice questions

Uses the shared qcm_accuracy module for consistent accuracy calculation
across all evaluators and trainers.
"""

from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
from PIL import Image

from .base_evaluator import BaseEvaluator
from .qcm_accuracy import extract_answer_letter, calculate_qcm_accuracy, normalize_text

logger = logging.getLogger(__name__)


class QCMEvaluator(BaseEvaluator):
    """Evaluator for QCM (Multiple Choice Questions) on ERP screenshots"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_path = None
        self.image_dir = None

    def evaluate(self, model_path: str = None, max_samples: int = None,
                 dataset_path: str = None, image_dir: str = None) -> Dict[str, Any]:
        """Evaluate on QCM dataset"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        if dataset_path:
            self.dataset_path = Path(dataset_path)
        if image_dir:
            self.image_dir = Path(image_dir)

        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError(f"QCM dataset not found: {self.dataset_path}")

        logger.info(f"Evaluating on QCM dataset: {self.dataset_path}")

        # Load QCM dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle nested structure if present
        if raw_data and 'qcm' in raw_data[0]:
            dataset = [(item['qcm'], item.get('image_name', '')) for item in raw_data]
        else:
            dataset = [(item, item.get('image_name', '')) for item in raw_data]

        if max_samples:
            dataset = dataset[:max_samples]

        results = []
        for qcm_data, image_name in tqdm(dataset, desc="QCM Evaluation"):
            # Load image
            if image_name and self.image_dir:
                image_path = self.image_dir / image_name
                if image_path.exists():
                    image = self.load_image(str(image_path))
                else:
                    image = Image.new('RGB', (224, 224), color='white')
            else:
                image = Image.new('RGB', (224, 224), color='white')

            # Format question with options
            question = qcm_data['question']
            options = qcm_data['options']
            correct_answer = qcm_data['correct_answer']

            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = f"{question}\n\nOptions:\n{options_text}\n\nFirst, state the letter of the correct answer. YOU MUST OUTPUT THE CORRECT LETTER FIRST, then the text of the answer, then provide your explanation.\n\nAnswer:"

            response = self.generate_response(image, prompt)

            # Extract predicted answer using shared extractor
            predicted_letter = extract_answer_letter(response, list(options.keys()))

            results.append({
                "question": question,
                "options": options,
                "response": response,
                "predicted_letter": predicted_letter,
                "correct_answer": correct_answer,
                "is_correct": predicted_letter == correct_answer,
                "dataset": "erp_qcm"
            })

        # Use shared accuracy calculation
        metrics = calculate_qcm_accuracy(results, split="full")
        accuracy = metrics["accuracy"]

        return {
            "benchmark": "erp_qcm",
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct": sum(1 for r in results if r['is_correct']),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate QCM accuracy using the shared qcm_accuracy module.
        This implements the abstract method from BaseEvaluator.
        """
        if not results:
            return 0.0
        metrics = calculate_qcm_accuracy(results, split="full")
        return metrics["accuracy"]