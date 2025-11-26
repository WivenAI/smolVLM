#!/usr/bin/env python3
"""
Sanity check: Evaluate trained model on the SAME samples used for training.
This proves whether the training mechanism works correctly.
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
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
from tqdm import tqdm
import argparse


def load_training_samples(benchmark_name: str, max_samples: int, seed: int = 42):
    """Load the EXACT same samples that were used for training"""
    print(f"Loading {benchmark_name} dataset (matching training data)...")

    # Load OCRBench exactly as in training
    if benchmark_name == "ocrbench":
        try:
            dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        except:
            try:
                dataset = load_dataset("lmms-lab/OCRBench-v2", split="test", trust_remote_code=True)
            except:
                raise ValueError("Could not load OCRBench dataset")
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Apply SAME random sampling as training (with seed for reproducibility)
    if max_samples and len(dataset) > max_samples:
        import random
        random.seed(seed)  # Use seed to get same samples
        indices = random.sample(range(len(dataset)), max_samples)
        indices.sort()  # Sort for reproducibility
        dataset = dataset.select(indices)

    print(f"Loaded {len(dataset)} samples (matching training set)")
    return dataset


def evaluate_on_samples(model, processor, dataset, device):
    """Evaluate model on given samples"""
    model.eval()

    results = []
    correct = 0
    total = 0

    print("\nEvaluating on training samples...")

    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]

        # Extract image
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            continue

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract question
        if 'query' in item:
            if isinstance(item['query'], dict):
                question = item['query'].get('en', '')
            else:
                question = item['query']
        elif 'question' in item:
            question = item['question']
        else:
            question = "What do you see in this image?"

        # Extract ground truth answer
        if 'answers' in item:
            answers = item['answers']
            if isinstance(answers, list) and len(answers) > 0:
                ground_truth = answers[0]
            else:
                ground_truth = str(answers)
        elif 'answer' in item:
            ground_truth = item['answer']
        elif 'label' in item:
            ground_truth = str(item['label'])
        else:
            ground_truth = "Unknown"

        # Ensure ground_truth is a string (can be list in some datasets)
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if len(ground_truth) > 0 else "Unknown"
        ground_truth = str(ground_truth)

        # Prepare input using chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # Process with image
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Decode response
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Extract assistant response (after last "Assistant:" or similar)
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        elif "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
            response = response.lstrip(":").strip()
        else:
            response = generated_text.strip()

        # Check if correct (simple string matching)
        is_correct = ground_truth.lower().strip() in response.lower().strip() or \
                     response.lower().strip() in ground_truth.lower().strip()

        if is_correct:
            correct += 1
        total += 1

        # Save result
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "correct": is_correct
        })

        # Show first few examples
        if idx < 5:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Response: {response}")
            print(f"Correct: {is_correct}")

    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"SANITY CHECK RESULTS (Training Data)")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}")

    if accuracy < 50:
        print("\n⚠️  WARNING: Accuracy is low even on training data!")
        print("This suggests the training mechanism may have issues.")
    elif accuracy > 80:
        print("\n✅ Training mechanism appears to work correctly!")
        print("The model learned from the training data.")
    else:
        print("\n⚡ Moderate accuracy on training data.")
        print("Training may be working but not overfitting as expected.")

    return results, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check: Evaluate on training samples"
    )
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--benchmark", type=str, default="ocrbench",
                       help="Benchmark name (must match training)")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Number of samples (must match training)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sample selection")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    # Set output path
    if args.output is None:
        model_name = Path(args.model_path).name
        args.output = f"sanity_check_{model_name}.json"

    print(f"Sanity Check Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Samples: {args.max_samples}")
    print(f"Seed: {args.seed}")

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("\nLoading trained model...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # Load training samples
    dataset = load_training_samples(args.benchmark, args.max_samples, args.seed)

    # Evaluate
    results, accuracy = evaluate_on_samples(model, processor, dataset, device)

    # Save results
    output_data = {
        "model_path": args.model_path,
        "benchmark": args.benchmark,
        "num_samples": len(dataset),
        "seed": args.seed,
        "accuracy": accuracy,
        "results": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
