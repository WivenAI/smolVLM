#!/usr/bin/env python3
"""
Quick test of BERTScore benchmark on 2 samples
"""

import json
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from bert_score import score as bert_score


def test_bertscore():
    print("=" * 80)
    print("TESTING BERTSCORE BENCHMARK - 2 SAMPLES")
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
    print(f"   Model loaded: {model_path}")

    # Test on 2 samples
    print("\n3. Running BERTScore on 2 samples...")
    for i in range(2):
        item = dataset[i]
        print(f"\n   Sample {i+1}:")
        print(f"   - Image: {item['image_name']}")
        print(f"   - Type: {item['type']}")

        # Load image
        image_path = Path('dpo_image_dataset') / item['image_name']
        image = Image.open(image_path).convert('RGB')

        # Generate prediction
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": item['prompt']}
            ]
        }]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reference = item['chosen']

        # Calculate BERTScore
        P, R, F1 = bert_score([prediction], [reference], lang="fr", verbose=False, device=model.device)

        print(f"   - Prediction length: {len(prediction)} chars")
        print(f"   - Reference length: {len(reference)} chars")
        print(f"   - BERTScore F1: {F1[0]:.4f}")
        print(f"   - BERTScore Precision: {P[0]:.4f}")
        print(f"   - BERTScore Recall: {R[0]:.4f}")

    print("\n" + "=" * 80)
    print("✅ BERTSCORE BENCHMARK TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    test_bertscore()
