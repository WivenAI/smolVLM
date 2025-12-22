"""
DPO Log Probability Evaluator
Evaluates model preference alignment by comparing log probabilities of chosen vs rejected responses
"""

from typing import Dict, Any, List
from tqdm import tqdm
import logging
import torch
import torch.nn.functional as F
from PIL import Image

from .base_evaluator import BaseEvaluator
from .dpo_utils import load_dpo_dataset, DPODatasetIterator, ensure_model_loaded

logger = logging.getLogger(__name__)


class LogProbEvaluator(BaseEvaluator):
    """Evaluator for DPO preference alignment using log probabilities"""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)

    def compute_response_logprob(self, image: Image.Image, prompt: str, response: str) -> Dict[str, float]:
        """
        Compute the log probability of a response given an image and prompt

        Returns:
            Dictionary with total_logprob, avg_logprob, perplexity, num_tokens
        """
        # Format the full text with prompt and response
        full_text = f"<image>{prompt}\n{response}"

        # Process inputs
        inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Get device from model's first parameter (handles device_map="auto")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get input_ids for the full sequence
        input_ids = inputs['input_ids']

        # Get the prompt-only encoding to find where response starts
        prompt_only_text = f"<image>{prompt}\n"
        prompt_inputs = self.processor(
            text=prompt_only_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )
        prompt_inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in prompt_inputs.items()}

        prompt_length = prompt_inputs['input_ids'].shape[1]

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Ensure shift_labels is on same device as log_probs
        shift_labels = shift_labels.to(log_probs.device)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Only consider tokens from the response part (after prompt)
        response_start_idx = max(0, prompt_length - 1)
        response_log_probs = token_log_probs[:, response_start_idx:]

        # Calculate metrics
        total_logprob = response_log_probs.sum().item()
        num_tokens = response_log_probs.shape[1]
        avg_logprob = total_logprob / num_tokens if num_tokens > 0 else 0.0
        perplexity = torch.exp(-response_log_probs.mean()).item() if num_tokens > 0 else float('inf')

        return {
            'total_logprob': total_logprob,
            'avg_logprob': avg_logprob,
            'perplexity': perplexity,
            'num_tokens': num_tokens
        }

    def evaluate(self, dataset_path: str = None, image_dir: str = None,
                 model_path: str = None, max_samples: int = None,
                 use_fixed_subset: bool = False, subset_seed: int = 42) -> Dict[str, Any]:
        """Evaluate log probabilities for chosen vs rejected responses

        Args:
            dataset_path: Path to the DPO dataset JSON file
            image_dir: Directory containing images
            model_path: Path to model weights (optional)
            max_samples: Maximum number of samples to evaluate
            use_fixed_subset: If True, use a fixed random subset for consistent evaluation
            subset_seed: Seed for reproducible subset selection (default: 42)
        """
        ensure_model_loaded(self, model_path)
        logger.info("Evaluating DPO log probabilities...")

        # Load dataset using shared utility
        dataset = load_dpo_dataset(dataset_path, max_samples, use_fixed_subset, subset_seed)
        logger.info(f"Loaded {len(dataset)} DPO examples")

        # Initialize result collectors
        results = []
        chosen_logprobs = []
        rejected_logprobs = []
        chosen_perplexities = []
        rejected_perplexities = []
        margins = []
        preferences_correct = 0

        # Iterate using shared iterator
        iterator = DPODatasetIterator(dataset, image_dir, "LogProb")

        for item, image in tqdm(iterator, desc="LogProb", total=len(dataset)):
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']

            # Compute log probabilities
            chosen_metrics = self.compute_response_logprob(image, prompt, chosen)
            rejected_metrics = self.compute_response_logprob(image, prompt, rejected)

            # Calculate margin (chosen should have higher log prob)
            margin = chosen_metrics['avg_logprob'] - rejected_metrics['avg_logprob']
            is_correct = margin > 0

            chosen_logprobs.append(chosen_metrics['avg_logprob'])
            rejected_logprobs.append(rejected_metrics['avg_logprob'])
            chosen_perplexities.append(chosen_metrics['perplexity'])
            rejected_perplexities.append(rejected_metrics['perplexity'])
            margins.append(margin)

            if is_correct:
                preferences_correct += 1

            results.append({
                "image_name": item['image_name'],
                "chosen_logprob": chosen_metrics['avg_logprob'],
                "rejected_logprob": rejected_metrics['avg_logprob'],
                "chosen_perplexity": chosen_metrics['perplexity'],
                "rejected_perplexity": rejected_metrics['perplexity'],
                "margin": margin,
                "preference_correct": is_correct
            })

        # Log skip summary
        iterator.log_skip_summary()

        # Calculate overall metrics
        num_examples = len(results)
        accuracy = (preferences_correct / num_examples * 100) if num_examples > 0 else 0.0

        # Filter out infinite perplexities for mean calculation
        valid_chosen_ppl = [p for p in chosen_perplexities if p != float('inf')]
        valid_rejected_ppl = [p for p in rejected_perplexities if p != float('inf')]

        return {
            "benchmark": "dpo_logprob",
            "accuracy": accuracy,
            "total_samples": num_examples,
            "preferences_correct": preferences_correct,
            "chosen_avg_logprob": sum(chosen_logprobs) / len(chosen_logprobs) if chosen_logprobs else 0.0,
            "rejected_avg_logprob": sum(rejected_logprobs) / len(rejected_logprobs) if rejected_logprobs else 0.0,
            "chosen_perplexity": sum(valid_chosen_ppl) / len(valid_chosen_ppl) if valid_chosen_ppl else 0.0,
            "rejected_perplexity": sum(valid_rejected_ppl) / len(valid_rejected_ppl) if valid_rejected_ppl else 0.0,
            "margin_mean": sum(margins) / len(margins) if margins else 0.0,
            "results": results
        }

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate preference accuracy from results"""
        if not results:
            return 0.0
        correct = sum(1 for r in results if r.get('preference_correct', False))
        return (correct / len(results) * 100)
