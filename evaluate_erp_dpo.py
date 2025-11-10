#!/usr/bin/env python3
"""
ERP DPO Evaluation Script - BERTScore and Log Probability

Evaluates models on ERP DPO dataset with metrics:
1. BERTScore - Semantic similarity between model response and chosen response
2. Log Probability - Model's preference for chosen vs rejected responses
3. Preference Accuracy - Does model assign higher probability to chosen?

Used for evaluating DPO-style datasets with chosen/rejected pairs.
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ERPDPOEvaluator:
    """Evaluator for ERP DPO dataset with BERTScore and log probability metrics"""

    def __init__(
        self,
        model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        dataset_path: str = "dpo_image_dataset/dpo_dataset_gemini.json",
        image_dir: str = "dpo_image_dataset"
    ):
        """Initialize the evaluator

        Args:
            model_path: Path to model (base or fine-tuned)
            dataset_path: Path to DPO dataset JSON
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

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_dataset(self) -> List[Dict]:
        """Load and parse DPO dataset

        Returns:
            List of samples with prompt, chosen, rejected, image
        """
        logger.info(f"Loading dataset: {self.dataset_path}")

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} samples")
        return data

    def load_image(self, image_name: Optional[str]) -> Optional[Image.Image]:
        """Load an image from the image directory

        Args:
            image_name: Name of the image file

        Returns:
            PIL Image or None if not found
        """
        if image_name is None:
            return None

        # Try different paths
        possible_paths = [
            self.image_dir / image_name,
            self.image_dir / "images" / image_name,
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
        max_new_tokens: int = 200
    ) -> str:
        """Generate response from model

        Args:
            prompt: Text prompt
            image: Optional image
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated response text
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
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

            generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        return response.strip()

    def compute_response_logprob(
        self,
        prompt: str,
        response: str,
        image: Optional[Image.Image] = None
    ) -> Dict[str, float]:
        """Compute log probability of a response given prompt and image

        Args:
            prompt: Input prompt
            response: Target response to score
            image: Optional image

        Returns:
            Dict with log probability metrics
        """
        # Prepare full conversation
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        if image is not None:
            messages[0]["content"].append({"type": "image"})

        messages[0]["content"].append({"type": "text", "text": prompt})

        # Add assistant response
        # Format as list to match user message structure
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        # Apply chat template
        full_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False
        )

        # Process inputs
        inputs = self.processor(
            text=full_text,
            images=[image] if image is not None else None,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        input_ids = inputs['input_ids']

        # Compute log probabilities
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # We want log prob of the response tokens only
        # Find where assistant response starts (after the last user message)
        prompt_only_messages = messages[:1]  # Just user message
        prompt_only_text = self.processor.apply_chat_template(
            prompt_only_messages,
            add_generation_prompt=True
        )

        prompt_inputs = self.processor(
            text=prompt_only_text,
            images=[image] if image is not None else None,
            return_tensors="pt"
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]

        # Response tokens start after prompt
        response_start_idx = max(0, prompt_length - 1)
        response_log_probs = token_log_probs[:, response_start_idx:]

        # Calculate metrics
        total_logprob = response_log_probs.sum().item()
        num_tokens = response_log_probs.shape[1]
        avg_logprob = total_logprob / num_tokens if num_tokens > 0 else float('-inf')

        return {
            'total_logprob': total_logprob,
            'avg_logprob': avg_logprob,
            'num_tokens': num_tokens
        }

    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
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
        logger.info("Starting ERP DPO Evaluation")
        logger.info("="*80)

        # Load model and dataset
        self.load_model()
        samples = self.load_dataset()

        if max_samples:
            samples = samples[:max_samples]
            logger.info(f"Evaluating on {len(samples)} samples (limited)")

        # Evaluate each sample
        results = []
        generated_responses = []
        chosen_responses = []

        chosen_logprobs = []
        rejected_logprobs = []

        for sample in tqdm(samples, desc="Evaluating"):
            prompt = sample['prompt']
            chosen = sample['chosen']
            rejected = sample['rejected']
            image_name = sample.get('image_name', None)

            # Load image if available
            image = self.load_image(image_name)

            # Generate model response
            generated = self.generate_response(prompt, image)

            # Compute log probabilities for chosen and rejected
            chosen_metrics = self.compute_response_logprob(prompt, chosen, image)
            rejected_metrics = self.compute_response_logprob(prompt, rejected, image)

            # Calculate preference margin
            margin = chosen_metrics['avg_logprob'] - rejected_metrics['avg_logprob']
            prefers_chosen = margin > 0

            # Store result
            result = {
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'generated': generated,
                'chosen_avg_logprob': chosen_metrics['avg_logprob'],
                'rejected_avg_logprob': rejected_metrics['avg_logprob'],
                'margin': margin,
                'prefers_chosen': prefers_chosen,
                'image_name': image_name,
                'type': sample.get('type', None)
            }

            results.append(result)

            # For BERTScore
            if compute_bertscore:
                generated_responses.append(generated)
                chosen_responses.append(chosen)

            # Collect log probs
            chosen_logprobs.append(chosen_metrics['avg_logprob'])
            rejected_logprobs.append(rejected_metrics['avg_logprob'])

        # Calculate aggregate metrics
        avg_chosen_logprob = np.mean(chosen_logprobs)
        avg_rejected_logprob = np.mean(rejected_logprobs)
        avg_margin = avg_chosen_logprob - avg_rejected_logprob

        preference_accuracy = sum(1 for r in results if r['prefers_chosen']) / len(results) * 100

        # Calculate BERTScore
        if compute_bertscore:
            bertscore_metrics = self.calculate_bertscore(
                generated_responses,
                chosen_responses
            )
        else:
            bertscore_metrics = None

        # Compile final results
        evaluation_results = {
            'model_path': self.model_path,
            'dataset_path': self.dataset_path,
            'num_samples': len(results),
            'metrics': {
                'avg_chosen_logprob': avg_chosen_logprob,
                'avg_rejected_logprob': avg_rejected_logprob,
                'avg_margin': avg_margin,
                'preference_accuracy': preference_accuracy,
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
        logger.info(f"\nPreference Accuracy: {preference_accuracy:.2f}%")
        logger.info(f"  (Model assigns higher probability to 'chosen' response)")
        logger.info(f"\nAverage Log Probabilities:")
        logger.info(f"  Chosen:   {avg_chosen_logprob:.4f}")
        logger.info(f"  Rejected: {avg_rejected_logprob:.4f}")
        logger.info(f"  Margin:   {avg_margin:.4f}")

        if bertscore_metrics:
            logger.info(f"\nBERTScore (Generated vs Chosen):")
            logger.info(f"  Precision: {bertscore_metrics['precision']:.4f}")
            logger.info(f"  Recall:    {bertscore_metrics['recall']:.4f}")
            logger.info(f"  F1:        {bertscore_metrics['f1']:.4f}")

        logger.info("="*80)

        return evaluation_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on ERP DPO dataset with BERTScore and log probabilities"
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
        default="dpo_image_dataset/dpo_dataset_gemini.json",
        help="Path to DPO dataset JSON"
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
        default="erp_dpo_evaluation.json",
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
    evaluator = ERPDPOEvaluator(
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
