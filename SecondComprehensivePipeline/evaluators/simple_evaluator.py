"""
Simple Evaluator - Minimal evaluator for sample generation and inference
"""

from .base_evaluator import BaseEvaluator
from typing import Dict, Any


class SimpleEvaluator(BaseEvaluator):
    """
    Simple evaluator that implements required abstract methods
    but is primarily used for generate() functionality for sample outputs.
    """

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Not used for sample generation"""
        raise NotImplementedError("SimpleEvaluator is only for generation, not evaluation")

    def calculate_accuracy(self, predictions: list, references: list) -> float:
        """Not used for sample generation"""
        raise NotImplementedError("SimpleEvaluator is only for generation, not evaluation")

    def generate(self, prompt: str, image=None, max_new_tokens: int = 256) -> str:
        """
        Generate response for a prompt and optional image.
        Wrapper around generate_response for compatibility with pipeline sample generation.

        Args:
            prompt: The question/prompt text
            image: Optional PIL Image
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        return self.generate_response(image, prompt, max_new_tokens)
