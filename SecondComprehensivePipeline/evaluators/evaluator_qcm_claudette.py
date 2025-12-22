"""
QCM Claudette Evaluator - Evaluates ERP multiple choice questions without images (text-only mode)

SmolVLM can "function as a pure language model without visual inputs" so we use text-only
mode for Claudette evaluation to test the model's knowledge without image context.

Uses the shared qcm_accuracy module for consistent accuracy calculation
across all evaluators and trainers.
"""

from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
import torch
from PIL import Image

from .base_evaluator import BaseEvaluator
from .qcm_accuracy import extract_answer_letter, calculate_qcm_accuracy, normalize_text

logger = logging.getLogger(__name__)


class QCMClaudetteEvaluator(BaseEvaluator):
    """Evaluator for QCM Claudette dataset (text-only mode, no images)"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self.dataset_path = None

    def generate_response_text_only(self, question: str, max_new_tokens: int = 256) -> str:
        """Generate response for text-only input (no image)

        SmolVLM supports functioning as a pure language model without visual inputs.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Text-only message format (no image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process without image
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode response
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def evaluate(self, model_path: str = None, max_samples: int = None,
                 dataset_path: str = None) -> Dict[str, Any]:
        """Evaluate on QCM Claudette dataset (text-only, no images)"""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_base_model()

        if dataset_path:
            self.dataset_path = Path(dataset_path)

        if not self.dataset_path or not self.dataset_path.exists():
            raise ValueError(f"QCM Claudette dataset not found: {self.dataset_path}")

        logger.info(f"Evaluating on QCM Claudette dataset (text-only mode): {self.dataset_path}")

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

        results = []
        for qcm_data in tqdm(dataset, desc="QCM Claudette (text-only)"):
            # Format question with options
            question = qcm_data['question']
            options = qcm_data['options']
            correct_answer = qcm_data['correct_answer']

            options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
            prompt = f"{question}\n\nOptions:\n{options_text}\n\nFirst, state the letter of the correct answer. YOU MUST OUTPUT THE CORRECT LETTER FIRST, then the text of the answer, then provide your explanation.\n\nAnswer:"

            # Use text-only generation (no image)
            response = self.generate_response_text_only(prompt)

            # Extract predicted answer using shared extractor
            predicted_letter = extract_answer_letter(response, list(options.keys()))

            results.append({
                "question": question,
                "options": options,
                "response": response,
                "predicted_letter": predicted_letter,
                "correct_answer": correct_answer,
                "is_correct": predicted_letter == correct_answer,
                "dataset": "qcm_claudette"
            })

        # Use shared accuracy calculation
        metrics = calculate_qcm_accuracy(results, split="full")
        accuracy = metrics["accuracy"]

        return {
            "benchmark": "qcm_claudette",
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
