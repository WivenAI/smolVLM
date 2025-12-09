"""
QCM Claudette Evaluator - Evaluates ERP multiple choice questions with black placeholder images
"""

from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
import re
from PIL import Image

from .base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class QCMClaudetteEvaluator(BaseEvaluator):
    """Evaluator for QCM Claudette dataset (uses black placeholder images)"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_path = None

    def evaluate(self, model_path: str = None, max_samples: int = None,
                 dataset_path: str = None) -> Dict[str, Any]:
        """Evaluate on QCM Claudette dataset with black images"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        if dataset_path:
            self.dataset_path = Path(dataset_path)

        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError(f"QCM Claudette dataset not found: {self.dataset_path}")

        logger.info(f"Evaluating on QCM Claudette dataset: {self.dataset_path}")

        # Load QCM dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle nested structure if present
        if raw_data and 'qcm' in raw_data[0]:
            dataset = [item['qcm'] for item in raw_data]
        else:
            dataset = raw_data

        if max_samples:
            dataset = dataset[:max_samples]

        # Create black placeholder image (used for all questions)
        black_image = Image.new('RGB', (224, 224), color='black')

        results = []
        for qcm_data in tqdm(dataset, desc="QCM Claudette"):
            # Format question with options
            question = qcm_data['question']
            options = qcm_data['options']
            correct_answer = qcm_data['correct_answer']

            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = f"{question}\n\nOptions:\n{options_text}\n\nFirst, state the letter of the correct answer. YOU MUST OUTPUT THE CORRECT LETTER FIRST, then the text of the answer, then provide your explanation.\n\nAnswer:"

            response = self.generate_response(black_image, prompt)

            # Extract predicted answer (letter)
            predicted_letter = self._extract_answer_letter(response, list(options.keys()))

            results.append({
                "question": question,
                "options": options,
                "response": response,
                "predicted_letter": predicted_letter,
                "correct_answer": correct_answer,
                "is_correct": predicted_letter == correct_answer,
                "dataset": "qcm_claudette"
            })

        accuracy = self.calculate_accuracy(results)

        return {
            "benchmark": "qcm_claudette",
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct": sum(1 for r in results if r['is_correct']),
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate QCM accuracy"""
        if not results:
            return 0.0

        correct = sum(1 for r in results if r.get('is_correct', False))
        total = len(results)

        return (correct / total * 100) if total > 0 else 0.0

    def _extract_answer_letter(self, response: str, valid_options: List[str]) -> str:
        """Extract the answer letter from the response"""
        response = response.strip().upper()

        # Check if response starts with a valid option letter
        for opt in valid_options:
            if response.startswith(opt.upper()):
                return opt.upper()

        # Look for pattern like "A)" or "A:" or "A -" or just "A"
        for opt in valid_options:
            pattern = rf'\b{opt.upper()}\b'
            if re.search(pattern, response):
                return opt.upper()

        # If no match found, return the first character if it's a valid option
        if response and response[0] in [o.upper() for o in valid_options]:
            return response[0]

        return ""
