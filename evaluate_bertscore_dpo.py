#!/usr/bin/env python3
"""
BERTScore Benchmark for DPO Image Dataset
Evaluates SmolVLM model responses against ground truth using BERTScore metrics
"""

import os
import json
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVision2Seq
from bert_score import score as bert_score


def load_model_and_processor(model_path="HuggingFaceTB/SmolVLM-500M-Instruct"):
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

    return model, processor


def generate_response(model, processor, image, prompt, max_new_tokens=512):
    """Generate response from the model"""
    # Format the input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Apply chat template
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process inputs
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
        size={"longest_edge": 1024}
    ).to(model.device)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # Decode
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    return generated_texts[0]


def evaluate_bertscore(dataset_path, image_dir, model, processor, output_file="bertscore_results.json"):
    """Evaluate the model using BERTScore on the DPO dataset"""

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
            "model": "SmolVLM-500M-Instruct"
        },
        "overall_metrics": {},
        "per_example_results": []
    }

    # Lists to store all predictions and references for overall metrics
    all_predictions = []
    all_references = []

    # Evaluate each example
    print("\nGenerating predictions and evaluating...")
    for idx, item in enumerate(tqdm(dataset)):
        # Load image
        image_path = Path(image_dir) / item['image_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize large images to avoid processor errors
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Generate prediction
        prompt = item['prompt']
        prediction = generate_response(model, processor, image, prompt)
        reference = item['chosen']

        # Store for overall metrics
        all_predictions.append(prediction)
        all_references.append(reference)

        # Calculate BERTScore for this example
        P, R, F1 = bert_score(
            [prediction],
            [reference],
            lang="fr",  # French language
            verbose=False,
            device=model.device
        )

        # Store individual result
        example_result = {
            "id": idx,
            "image_name": item['image_name'],
            "type": item['type'],
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "rejected": item['rejected'],
            "bertscore": {
                "precision": float(P[0]),
                "recall": float(R[0]),
                "f1": float(F1[0])
            }
        }

        results["per_example_results"].append(example_result)

    # Calculate overall BERTScore
    print("\nCalculating overall BERTScore metrics...")
    P_overall, R_overall, F1_overall = bert_score(
        all_predictions,
        all_references,
        lang="fr",
        verbose=True,
        device=model.device
    )

    results["overall_metrics"] = {
        "precision": {
            "mean": float(P_overall.mean()),
            "std": float(P_overall.std())
        },
        "recall": {
            "mean": float(R_overall.mean()),
            "std": float(R_overall.std())
        },
        "f1": {
            "mean": float(F1_overall.mean()),
            "std": float(F1_overall.std())
        }
    }

    # Save results
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*50)
    print("BERTSCORE EVALUATION RESULTS")
    print("="*50)
    print(f"Total examples evaluated: {len(dataset)}")
    print(f"\nOverall Metrics:")
    print(f"  Precision: {results['overall_metrics']['precision']['mean']:.4f} ± {results['overall_metrics']['precision']['std']:.4f}")
    print(f"  Recall:    {results['overall_metrics']['recall']['mean']:.4f} ± {results['overall_metrics']['recall']['std']:.4f}")
    print(f"  F1 Score:  {results['overall_metrics']['f1']['mean']:.4f} ± {results['overall_metrics']['f1']['std']:.4f}")
    print("="*50)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BERTScore Benchmark for DPO Dataset")
    parser.add_argument("--model-path", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct",
                       help="Path to model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--dataset-path", type=str, default="dpo_image_dataset/dpo_dataset_gemini.json",
                       help="Path to DPO dataset JSON")
    parser.add_argument("--image-dir", type=str, default="dpo_image_dataset",
                       help="Directory containing images")
    parser.add_argument("--output-file", type=str, default="dpo_bertscore_results.json",
                       help="Output file for results")

    args = parser.parse_args()

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, processor = load_model_and_processor(args.model_path)

    # Run evaluation
    results = evaluate_bertscore(
        dataset_path=args.dataset_path,
        image_dir=args.image_dir,
        model=model,
        processor=processor,
        output_file=args.output_file
    )

    print(f"\nBERTScore evaluation completed!")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
