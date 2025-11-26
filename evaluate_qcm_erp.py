#!/usr/bin/env python3
"""
ERP QCM Benchmark Evaluation for SmolVLM
Evaluates the model on French ERP multiple-choice questions
"""

# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

import json
import torch
import argparse
import random
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, List, Any
import logging
import re
from PIL import Image
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERPQCMEvaluator:
    def __init__(self, model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct"):
        """Initialize the ERP QCM evaluator

        Args:
            model_path: Path to the model (defaults to base model)
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a dummy white image for text-only questions (SmolVLM requires an image)
        self.dummy_image = Image.new('RGB', (224, 224), color='white')

        # Set random seeds for reproducibility
        random.seed(42)
        torch.manual_seed(42)

    def load_model(self):
        """Load the model and processor"""
        logger.info(f"Loading model from: {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        logger.info("Model loaded successfully")

    def format_question(self, qcm_item: Dict) -> str:
        """Format a QCM question with options"""
        question = qcm_item['question']
        options = qcm_item['options']

        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])

        prompt = f"""Question: {question}

Options:
{options_text}

Répondez uniquement avec la lettre de la réponse correcte (A, B, C ou D):"""

        return prompt

    def generate_response(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate response for a text prompt (uses dummy white image since SmolVLM requires an image)"""
        try:
            # SmolVLM requires an image input, so we use a dummy white image for text-only questions
            # Format prompt with image token
            formatted_prompt = f"<image>{prompt}"

            inputs = self.processor(
                text=formatted_prompt,
                images=self.dummy_image,
                return_tensors="pt",
                size={"longest_edge": 224}
            ).to(self.device)

            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            # Clean up response - remove the prompt part
            response = generated_text.replace(prompt, "").strip()

            logger.debug(f"Generated: {response[:100]}")

            return response if response else "No answer"

        except Exception as e:
            logger.error(f"Error: {e}")
            return "Error"

    def extract_answer(self, response: str) -> str:
        """Extract answer letter (A, B, C, or D) from model response"""
        # Look for patterns like "A", "A.", "A)", "Answer: A", etc.
        patterns = [
            r'\b([ABCD])\b',  # Single letter
            r'([ABCD])[.)]',   # Letter with punctuation
            r'(?:answer|réponse)[:\s]+([ABCD])',  # "Answer: A" or "Réponse: A"
        ]

        response_upper = response.upper()

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)

        # If no match, look for the first occurrence of A, B, C, or D
        for char in response_upper:
            if char in ['A', 'B', 'C', 'D']:
                return char

        return "UNKNOWN"

    def load_qcm_data(self, qcm_path: str) -> List[Dict]:
        """Load QCM data from JSON file"""
        logger.info(f"Loading QCM data from: {qcm_path}")

        with open(qcm_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} questions")
        return data

    def evaluate(self, qcm_path: str, num_samples: int = None, percentage: float = 100.0) -> Dict[str, Any]:
        """Evaluate model on QCM benchmark

        Args:
            qcm_path: Path to QCM JSON file
            num_samples: Number of samples to evaluate (None = all)
            percentage: Percentage of dataset to use (1-100)

        Returns:
            Dictionary with evaluation results
        """
        if self.model is None or self.processor is None:
            self.load_model()

        # Load QCM data
        qcm_data = self.load_qcm_data(qcm_path)

        # Sample data based on percentage
        if percentage < 100.0:
            sample_size = int(len(qcm_data) * (percentage / 100.0))
            qcm_data = random.sample(qcm_data, sample_size)
            logger.info(f"Using {percentage}% of data: {len(qcm_data)} questions")

        # Further limit by num_samples if specified
        if num_samples is not None and len(qcm_data) > num_samples:
            qcm_data = random.sample(qcm_data, num_samples)
            logger.info(f"Limited to {num_samples} samples")

        results = []
        correct = 0
        total = 0

        # Evaluate each question
        for item in tqdm(qcm_data, desc="Evaluating QCM"):
            try:
                # Format question
                prompt = self.format_question(item)

                # Generate response
                response = self.generate_response(prompt)

                # Extract answer
                predicted_answer = self.extract_answer(response)
                correct_answer = item['correct_answer']

                # Check if correct
                is_correct = (predicted_answer == correct_answer)
                if is_correct:
                    correct += 1
                total += 1

                # Store result
                results.append({
                    "id": item['id'],
                    "question": item['question'][:100] + "...",  # Truncate for storage
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "raw_response": response[:200],  # Truncate
                    "is_correct": is_correct
                })

            except Exception as e:
                logger.warning(f"Error processing question {item.get('id', 'unknown')}: {e}")
                continue

        accuracy = (correct / total * 100) if total > 0 else 0.0

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": accuracy,
            "results": results
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("ERP QCM Benchmark Evaluation Summary")
        print("="*80)

        print(f"\nTotal Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")

        # Show some examples
        print("\n" + "-"*80)
        print("Sample Results (first 5):")
        print("-"*80)

        for i, result in enumerate(results['results'][:5], 1):
            status = "✓" if result['is_correct'] else "✗"
            print(f"\n{status} Question {result['id']}:")
            print(f"  Q: {result['question']}")
            print(f"  Correct: {result['correct_answer']}")
            print(f"  Predicted: {result['predicted_answer']}")
            print(f"  Response: {result['raw_response'][:100]}...")

        print("\n" + "="*80)

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM on ERP QCM benchmark")
    parser.add_argument("--model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct",
                       help="Path to the model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--qcm-path", default="balanced_qcm_all.json",
                       help="Path to QCM JSON file")
    parser.add_argument("--output-file", default="erp_qcm_results.json",
                       help="Output file for results")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to evaluate (None = use all)")
    parser.add_argument("--percentage", type=float, default=100.0,
                       help="Percentage of dataset to use (1-100, default: 100)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate percentage
    if args.percentage <= 0 or args.percentage > 100:
        parser.error("--percentage must be between 1 and 100")

    # Initialize evaluator
    evaluator = ERPQCMEvaluator(args.model_path)

    try:
        # Run evaluation
        results = evaluator.evaluate(
            qcm_path=args.qcm_path,
            num_samples=args.num_samples,
            percentage=args.percentage
        )

        # Save results
        evaluator.save_results(results, args.output_file)

        # Print summary
        evaluator.print_summary(results)

        print(f"\nDetailed results saved to: {args.output_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
