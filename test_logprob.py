#!/usr/bin/env python3
"""
Quick test of Log Probability benchmark on 2 samples
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


def compute_response_logprob(model, processor, image, prompt, response):
    """Compute log probability of a response"""
    full_text = f"<image>{prompt}\n{response}"

    inputs = processor(text=full_text, images=image, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    input_ids = inputs['input_ids']

    # Get prompt length
    prompt_only_text = f"<image>{prompt}\n"
    prompt_inputs = processor(text=prompt_only_text, images=image, return_tensors="pt", padding=True).to(model.device)
    prompt_length = prompt_inputs['input_ids'].shape[1]

    # Compute log probs
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    response_start_idx = max(0, prompt_length - 1)
    response_log_probs = token_log_probs[:, response_start_idx:]

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


def test_logprob():
    print("=" * 80)
    print("TESTING LOG PROBABILITY BENCHMARK - 2 SAMPLES")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading dataset...")
    with open('dpo_image_dataset/dpo_dataset_gemini.json', 'r') as f:
        dataset = json.load(f)
    print(f"   Total examples: {len(dataset)}")
    print(f"   Testing on first 2 examples")

    # Load model
    print("\n2. Loading model...")
    model_path = "HuggingFaceTB/SmolVLM-500M-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"   Model loaded: {model_path}")

    # Test on 2 samples
    print("\n3. Computing log probabilities for 2 samples...")
    for i in range(2):
        item = dataset[i]
        print(f"\n   Sample {i+1}:")
        print(f"   - Image: {item['image_name']}")
        print(f"   - Type: {item['type']}")

        # Load image
        image_path = Path('dpo_image_dataset') / item['image_name']
        image = Image.open(image_path).convert('RGB')

        # Compute log probs for chosen and rejected
        chosen_metrics = compute_response_logprob(model, processor, image, item['prompt'], item['chosen'])
        rejected_metrics = compute_response_logprob(model, processor, image, item['prompt'], item['rejected'])

        margin = chosen_metrics['avg_logprob'] - rejected_metrics['avg_logprob']
        preference_correct = margin > 0

        print(f"   - Chosen avg logprob: {chosen_metrics['avg_logprob']:.4f} ({chosen_metrics['num_tokens']} tokens)")
        print(f"   - Rejected avg logprob: {rejected_metrics['avg_logprob']:.4f} ({rejected_metrics['num_tokens']} tokens)")
        print(f"   - Margin: {margin:.4f}")
        print(f"   - Preference correct: {preference_correct} {'✓' if preference_correct else '✗'}")
        print(f"   - Chosen perplexity: {chosen_metrics['perplexity']:.2f}")
        print(f"   - Rejected perplexity: {rejected_metrics['perplexity']:.2f}")

    print("\n" + "=" * 80)
    print("✅ LOG PROBABILITY BENCHMARK TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_logprob()
