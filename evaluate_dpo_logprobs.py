#!/usr/bin/env python3
"""
DPO Log Probability Benchmark for SmolVLM
Evaluates model preference alignment by comparing log probabilities of chosen vs rejected responses
"""

import os
import json
import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch.nn.functional as F


def load_model_and_processor(model_path):
    """Load the SmolVLM model and processor"""
    print(f"Loading model: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.eval()  # Set to evaluation mode

    return model, processor


def compute_response_logprob(model, processor, image, prompt, response):
    """
    Compute the log probability of a response given an image and prompt

    Returns:
        total_logprob: Sum of log probabilities for all tokens in the response
        avg_logprob: Average log probability per token
        perplexity: Perplexity of the response
    """
    # Format the full text with prompt and response
    full_text = f"<image>{prompt}\n{response}"

    # Process inputs
    inputs = processor(
        text=full_text,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)

    # Get input_ids for the full sequence
    input_ids = inputs['input_ids']

    # We need to compute log probs only for the response part
    # First, get the prompt-only encoding to find where response starts
    prompt_only_text = f"<image>{prompt}\n"
    prompt_inputs = processor(
        text=prompt_only_text,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    prompt_length = prompt_inputs['input_ids'].shape[1]

    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Only consider tokens from the response part (after prompt)
    # Adjust for the shift: prompt_length - 1 because of the shift
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


def evaluate_dpo_logprobs(dataset_path, image_dir, model, processor, output_file="dpo_logprob_results.json"):
    """Evaluate log probabilities for chosen vs rejected responses"""

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Dataset loaded: {len(dataset)} examples")

    # Storage for results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_path": dataset_path,
            "num_examples": len(dataset),
            "model": str(model.config._name_or_path) if hasattr(model.config, '_name_or_path') else "SmolVLM"
        },
        "overall_metrics": {},
        "per_example_results": []
    }

    # Metrics accumulators
    chosen_logprobs = []
    rejected_logprobs = []
    margins = []
    chosen_perplexities = []
    rejected_perplexities = []
    preferences_correct = 0

    # Evaluate each example
    print("\nComputing log probabilities...")
    for idx, item in enumerate(tqdm(dataset)):
        try:
            # Load image
            image_path = Path(image_dir) / item['image_name']
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize large images to avoid processor errors
            max_size = 1024
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Get prompt and responses
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']

            # Compute log probabilities
            chosen_metrics = compute_response_logprob(model, processor, image, prompt, chosen)
            rejected_metrics = compute_response_logprob(model, processor, image, prompt, rejected)

            # Calculate margin (chosen should have higher log prob)
            margin = chosen_metrics['avg_logprob'] - rejected_metrics['avg_logprob']
            is_correct = margin > 0  # Model prefers chosen over rejected

            # Store metrics
            chosen_logprobs.append(chosen_metrics['avg_logprob'])
            rejected_logprobs.append(rejected_metrics['avg_logprob'])
            margins.append(margin)
            chosen_perplexities.append(chosen_metrics['perplexity'])
            rejected_perplexities.append(rejected_metrics['perplexity'])

            if is_correct:
                preferences_correct += 1

            # Store individual result
            example_result = {
                "id": idx,
                "image_name": item['image_name'],
                "type": item['type'],
                "prompt": prompt,
                "chosen_metrics": chosen_metrics,
                "rejected_metrics": rejected_metrics,
                "margin": margin,
                "preference_correct": is_correct
            }

            results["per_example_results"].append(example_result)

        except Exception as e:
            print(f"\nError processing example {idx}: {e}")
            continue

    # Calculate overall metrics
    num_examples = len(margins)

    results["overall_metrics"] = {
        "preference_accuracy": preferences_correct / num_examples if num_examples > 0 else 0.0,
        "num_correct_preferences": preferences_correct,
        "num_examples": num_examples,
        "chosen_avg_logprob": {
            "mean": sum(chosen_logprobs) / len(chosen_logprobs) if chosen_logprobs else 0.0,
            "min": min(chosen_logprobs) if chosen_logprobs else 0.0,
            "max": max(chosen_logprobs) if chosen_logprobs else 0.0
        },
        "rejected_avg_logprob": {
            "mean": sum(rejected_logprobs) / len(rejected_logprobs) if rejected_logprobs else 0.0,
            "min": min(rejected_logprobs) if rejected_logprobs else 0.0,
            "max": max(rejected_logprobs) if rejected_logprobs else 0.0
        },
        "margin": {
            "mean": sum(margins) / len(margins) if margins else 0.0,
            "min": min(margins) if margins else 0.0,
            "max": max(margins) if margins else 0.0
        },
        "chosen_perplexity": {
            "mean": sum(chosen_perplexities) / len(chosen_perplexities) if chosen_perplexities else 0.0,
            "min": min(chosen_perplexities) if chosen_perplexities else float('inf'),
            "max": max(chosen_perplexities) if chosen_perplexities else float('inf')
        },
        "rejected_perplexity": {
            "mean": sum(rejected_perplexities) / len(rejected_perplexities) if rejected_perplexities else 0.0,
            "min": min(rejected_perplexities) if rejected_perplexities else float('inf'),
            "max": max(rejected_perplexities) if rejected_perplexities else float('inf')
        }
    }

    # Save results
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*80)
    print("DPO LOG PROBABILITY EVALUATION RESULTS")
    print("="*80)
    print(f"Total examples evaluated: {num_examples}")
    print(f"\nPreference Alignment:")
    print(f"  Accuracy: {results['overall_metrics']['preference_accuracy']:.2%}")
    print(f"  Correct: {preferences_correct}/{num_examples}")
    print(f"\nLog Probability (Average per Token):")
    print(f"  Chosen:   {results['overall_metrics']['chosen_avg_logprob']['mean']:.4f}")
    print(f"  Rejected: {results['overall_metrics']['rejected_avg_logprob']['mean']:.4f}")
    print(f"  Margin:   {results['overall_metrics']['margin']['mean']:.4f}")
    print(f"\nPerplexity:")
    print(f"  Chosen:   {results['overall_metrics']['chosen_perplexity']['mean']:.2f}")
    print(f"  Rejected: {results['overall_metrics']['rejected_perplexity']['mean']:.2f}")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description="DPO Log Probability Benchmark")
    parser.add_argument("--model-path", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct",
                       help="Path to model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--dataset-path", type=str, default="dpo_image_dataset/dpo_dataset.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--output-file", type=str, default="dpo_logprob_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Run evaluation
    results = evaluate_dpo_logprobs(
        dataset_path=args.dataset_path,
        image_dir=args.image_dir,
        model=model,
        processor=processor,
        output_file=args.output_file
    )

    print(f"\nDPO log probability evaluation completed!")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
