#!/usr/bin/env python3
"""
ERP QCM Evaluation Script - Comprehensive Metrics

Evaluates models on ERP QCM dataset with three key metrics:
1. Accuracy - Does the model select the correct answer?
2. BERTScore - Semantic similarity between response and correct answer
3. Logprob - Probability model assigns to the correct answer

Supports both nested and flat QCM dataset formats.
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from bert_score import score as bert_score
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERPQCMEvaluator:
    """Evaluator for ERP QCM dataset with comprehensive metrics"""

    def __init__(
        self,
        model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        dataset_path: str = "dpo_image_dataset/qcm/qcm_dataset.json",
        image_dir: str = "dpo_image_dataset"
    ):
        """Initialize the evaluator

        Args:
            model_path: Path to model (base or fine-tuned)
            dataset_path: Path to QCM dataset JSON
            image_dir: Directory containing images
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.image_dir = Path(image_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Image dir: {image_dir}")

    def load_model(self):
        """Load model and processor"""
        logger.info(f"Loading model from: {self.model_path}")

        try:
            from peft import PeftModel

            # Check if LoRA adapter
            model_path = Path(self.model_path)
            is_adapter = (model_path / "adapter_config.json").exists()

            if is_adapter:
                logger.info("Detected LoRA adapter, loading base model first...")
                base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                base_model_obj = AutoModelForVision2Seq.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )

                self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
                logger.info("LoRA adapter loaded successfully")
            else:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info("Full model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_dataset(self) -> List[Dict]:
        """Load and parse QCM dataset

        Returns:
            List of samples with standardized format
        """
        logger.info(f"Loading dataset: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            # Detect format: nested (has 'qcm' key) or flat
            if 'qcm' in item:
                # Nested format
                sample = {
                    'id': item.get('id', len(samples)),
                    'image_name': item.get('image_name', None),
                    'question': item['qcm']['question'],
                    'options': item['qcm']['options'],
                    'correct_answer': item['qcm']['correct_answer'],
                    'explanation': item['qcm'].get('explanation', None)
                }
            else:
                # Flat format
                sample = {
                    'id': item.get('id', len(samples)),
                    'image_name': item.get('image_name', None),
                    'question': item['question'],
                    'options': item['options'],
                    'correct_answer': item['correct_answer'],
                    'explanation': item.get('explanation', None)
                }

            samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    def format_qcm_prompt(self, sample: Dict) -> str:
        """Format a QCM question into a prompt

        Args:
            sample: QCM sample

        Returns:
            Formatted prompt string
        """
        question = sample['question']
        options = sample['options']

        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])

        prompt = f"""Question: {question}

Options:
{options_text}

Répondez uniquement avec la lettre de la réponse correcte (A, B, C, ou D)."""

        return prompt

    def load_image(self, image_name: Optional[str]) -> Optional[Image.Image]:
        """Load an image from the image directory

        Args:
            image_name: Name of the image file

        Returns:
            PIL Image or None if not found/needed
        """
        if image_name is None:
            return None

        # Try different paths
        possible_paths = [
            self.image_dir / image_name,
            self.image_dir / "images" / image_name,
            self.image_dir / "qcm" / image_name,
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    return Image.open(path).convert('RGB')
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")

        logger.warning(f"Image not found: {image_name}")
        return None

    def generate_response(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        compute_logprobs: bool = True
    ) -> Dict[str, Any]:
        """Generate response from model with optional logprob calculation

        Args:
            prompt: Text prompt
            image: Optional image
            compute_logprobs: Whether to compute log probabilities

        Returns:
            Dict with 'response' and optionally 'logprobs'
        """
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        if image is not None:
            messages[0]["content"].append({"type": "image"})

        messages[0]["content"].append({"type": "text", "text": prompt})

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=prompt_text,
            images=[image] if image is not None else None,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if compute_logprobs:
                # Generate with output_scores to get logprobs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                )

                generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]

                # Calculate average log probability
                if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                    # Stack scores for all generated tokens
                    logits_stack = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
                    log_probs = torch.nn.functional.log_softmax(logits_stack, dim=-1)

                    # Get log probs for actual generated tokens
                    token_log_probs = []
                    for i in range(generated_ids.shape[1]):
                        if i < log_probs.shape[1]:
                            token_id = generated_ids[0, i]
                            token_log_prob = log_probs[0, i, token_id].item()
                            token_log_probs.append(token_log_prob)

                    avg_log_prob = np.mean(token_log_probs) if token_log_probs else float('-inf')
                else:
                    avg_log_prob = float('-inf')

                # Decode response
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                return {
                    'response': response.strip(),
                    'avg_log_prob': avg_log_prob,
                    'token_log_probs': token_log_probs
                }
            else:
                # Simple generation without logprobs
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )

                generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

                return {'response': response.strip()}

    def extract_answer_letter(self, response: str) -> Optional[str]:
        """Extract answer letter (A, B, C, D) from response

        Args:
            response: Model response text

        Returns:
            Extracted letter or None
        """
        response = response.strip().upper()

        # Direct single letter
        if len(response) == 1 and response in ['A', 'B', 'C', 'D']:
            return response

        # Pattern: "A", "B.", "C)", etc.
        match = re.search(r'\b([ABCD])[\.\):]?\b', response)
        if match:
            return match.group(1)

        # Check first character
        if response and response[0] in ['A', 'B', 'C', 'D']:
            return response[0]

        return None

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate accuracy from results

        Args:
            results: List of evaluation results

        Returns:
            Accuracy percentage
        """
        if not results:
            return 0.0

        correct = sum(1 for r in results if r.get('is_correct', False))
        return (correct / len(results)) * 100

    def calculate_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore for predictions vs references

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dict with P, R, F1 scores
        """
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        logger.info("Calculating BERTScore...")

        # Calculate BERTScore
        P, R, F1 = bert_score(
            predictions,
            references,
            lang='fr',  # French
            verbose=False,
            device=self.device
        )

        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }

    def evaluate(
        self,
        max_samples: Optional[int] = None,
        compute_bertscore: bool = True
    ) -> Dict[str, Any]:
        """Run complete evaluation

        Args:
            max_samples: Maximum number of samples to evaluate
            compute_bertscore: Whether to compute BERTScore

        Returns:
            Evaluation results
        """
        logger.info("="*80)
        logger.info("Starting ERP QCM Evaluation")
        logger.info("="*80)

        # Load model and dataset
        self.load_model()
        samples = self.load_dataset()

        if max_samples:
            samples = samples[:max_samples]
            logger.info(f"Evaluating on {len(samples)} samples (limited)")

        # Evaluate each sample
        results = []
        predictions = []
        references = []

        for sample in tqdm(samples, desc="Evaluating"):
            # Prepare prompt
            prompt = self.format_qcm_prompt(sample)

            # Load image if available
            image = self.load_image(sample['image_name'])

            # Generate response
            output = self.generate_response(prompt, image, compute_logprobs=True)
            response = output['response']
            avg_log_prob = output.get('avg_log_prob', None)

            # Extract answer
            predicted_answer = self.extract_answer_letter(response)
            correct_answer = sample['correct_answer']
            is_correct = (predicted_answer == correct_answer)

            # Get full text of correct answer for BERTScore
            correct_answer_text = sample['options'].get(correct_answer, "")

            # Store result
            result = {
                'id': sample['id'],
                'question': sample['question'],
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'response': response,
                'is_correct': is_correct,
                'avg_log_prob': avg_log_prob,
                'image_name': sample['image_name']
            }

            results.append(result)

            # For BERTScore
            if compute_bertscore:
                predictions.append(response)
                references.append(correct_answer_text)

        # Calculate metrics
        accuracy = self.calculate_accuracy(results)

        # Calculate average log probability
        log_probs = [r['avg_log_prob'] for r in results if r['avg_log_prob'] is not None]
        avg_log_prob = np.mean(log_probs) if log_probs else None

        # Calculate BERTScore
        if compute_bertscore:
            bertscore_metrics = self.calculate_bertscore(predictions, references)
        else:
            bertscore_metrics = None

        # Compile final results
        evaluation_results = {
            'model_path': self.model_path,
            'dataset_path': self.dataset_path,
            'num_samples': len(results),
            'metrics': {
                'accuracy': accuracy,
                'avg_log_prob': avg_log_prob,
                'bertscore': bertscore_metrics
            },
            'detailed_results': results
        }

        # Print summary
        logger.info("="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Samples: {len(results)}")
        logger.info(f"\nAccuracy: {accuracy:.2f}%")

        if avg_log_prob is not None:
            logger.info(f"Average Log Probability: {avg_log_prob:.4f}")

        if bertscore_metrics:
            logger.info(f"\nBERTScore:")
            logger.info(f"  Precision: {bertscore_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {bertscore_metrics['recall']:.4f}")
            logger.info(f"  F1:        {bertscore_metrics['f1']:.4f}")

        logger.info("="*80)

        return evaluation_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on ERP QCM dataset with comprehensive metrics"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Path to model (base or fine-tuned)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dpo_image_dataset/qcm/qcm_dataset.json",
        help="Path to QCM dataset JSON"
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default="dpo_image_dataset",
        help="Directory containing images"
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="erp_qcm_evaluation.json",
        help="Output JSON file for results"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate"
    )

    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore calculation (faster)"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = ERPQCMEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset,
        image_dir=args.image_dir
    )

    # Run evaluation
    results = evaluator.evaluate(
        max_samples=args.max_samples,
        compute_bertscore=not args.no_bertscore
    )

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
