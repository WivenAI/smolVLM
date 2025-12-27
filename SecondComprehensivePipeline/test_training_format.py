"""
Test Training Format - Visualize how training data is formatted

This script shows exactly how the training data is processed:
1. Original QCM/DPO data
2. Formatted messages (user + assistant)
3. Prompt text (with chat template applied)
4. Full text (prompt + response)
5. Input IDs and Labels (showing masked tokens with -100)
6. What tokens the model is actually trained on

Usage:
    python test_training_format.py --dataset qcm_gemini --num-samples 3
    python test_training_format.py --dataset dpo_gemini --num-samples 3
"""

import json
import argparse
from pathlib import Path
from PIL import Image
import torch

# Set HuggingFace cache before imports
from config.setup import setup_hf_cache, BASE_MODEL
setup_hf_cache()

from transformers import AutoProcessor


def visualize_qcm_training_format(dataset_path: str, image_dir: str, processor, num_samples: int = 3):
    """Visualize how QCM data is formatted for training"""

    print("=" * 100)
    print("QCM TRAINING FORMAT VISUALIZATION")
    print("=" * 100)
    print()

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Handle nested structure if present
    if raw_data and 'qcm' in raw_data[0]:
        dataset = [(item['qcm'], item.get('image_name', '')) for item in raw_data]
    else:
        dataset = [(item, item.get('image_name', '')) for item in raw_data]

    dataset = dataset[:num_samples]

    for idx, (qcm_data, image_name) in enumerate(dataset, 1):
        print("\n" + "=" * 100)
        print(f"SAMPLE {idx}")
        print("=" * 100)

        # 1. Original data
        print("\n[1] ORIGINAL DATA:")
        print("-" * 100)
        question = qcm_data['question']
        options = qcm_data['options']
        correct_answer = qcm_data['correct_answer']

        print(f"Question: {question}")
        print(f"Options: {json.dumps(options, indent=2)}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Image: {image_name}")

        # Load image
        image = None
        if image_name and Path(image_dir).exists():
            img_path = Path(image_dir) / image_name
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                print(f"Image loaded: {img_path} (size: {image.size})")
            else:
                image = Image.new('RGB', (224, 224), color='white')
                print(f"Image not found, using blank image")
        else:
            image = Image.new('RGB', (224, 224), color='white')
            print(f"No image path, using blank image")

        # 2. Format question with options (as done in training)
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        prompt = f"{question}\n\nOptions:\n{options_text}\n\nFirst, state the letter of the correct answer. YOU MUST OUTPUT THE CORRECT LETTER FIRST, then the text of the answer, then provide your explanation.\n\nAnswer:"

        # Train to output just the letter (matching evaluation format)
        answer = correct_answer

        # 3. Create messages (as done in training)
        print("\n[2] FORMATTED MESSAGES:")
        print("-" * 100)
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        full_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        print("User Message:")
        print(json.dumps(user_message, indent=2))
        print("\nFull Messages (with assistant response):")
        print(json.dumps(full_messages, indent=2))

        # 4. Apply chat template
        print("\n[3] CHAT TEMPLATE APPLIED:")
        print("-" * 100)
        prompt_text = processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        print("Prompt Text (what model sees as input):")
        print(repr(prompt_text))
        print("\nFull Text (prompt + response for training):")
        print(repr(full_text))

        # 5. Tokenize
        print("\n[4] TOKENIZATION:")
        print("-" * 100)
        prompt_inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        full_inputs = processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        prompt_length = prompt_inputs["input_ids"].shape[1]
        full_length = full_inputs["input_ids"].shape[1]

        print(f"Prompt token length: {prompt_length}")
        print(f"Full sequence token length: {full_length}")
        print(f"Response token length: {full_length - prompt_length}")

        # 6. Create labels (mask prompt tokens)
        print("\n[5] LABELS (MASKING):")
        print("-" * 100)
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        print(f"Masked tokens (set to -100): {prompt_length} tokens")
        print(f"Trainable tokens: {full_length - prompt_length} tokens")
        print("\nLabel masking visualization:")
        print(f"  Prompt tokens (0 to {prompt_length-1}): MASKED with -100 (not trained)")
        print(f"  Response tokens ({prompt_length} to {full_length-1}): TRAINED")

        # 7. Show actual tokens
        print("\n[6] TOKEN-BY-TOKEN BREAKDOWN:")
        print("-" * 100)

        # Decode individual tokens
        input_ids = full_inputs["input_ids"][0]
        label_ids = labels[0]

        print(f"\n{'Token ID':<10} {'Token':<30} {'Label':<10} {'Trained?':<10}")
        print("-" * 70)

        # Show first 20 tokens
        max_tokens_to_show = min(20, len(input_ids))
        for i in range(max_tokens_to_show):
            token_id = input_ids[i].item()
            label_id = label_ids[i].item()
            token_text = processor.decode([token_id], skip_special_tokens=False)
            is_trained = "NO (masked)" if label_id == -100 else "YES"
            print(f"{token_id:<10} {repr(token_text):<30} {label_id:<10} {is_trained:<10}")

        if len(input_ids) > max_tokens_to_show:
            print(f"... ({len(input_ids) - max_tokens_to_show} more tokens)")

        # Show last 10 tokens (should be the response)
        print("\nLast 10 tokens (should be the response being trained):")
        print(f"{'Token ID':<10} {'Token':<30} {'Label':<10} {'Trained?':<10}")
        print("-" * 70)
        for i in range(max(0, len(input_ids) - 10), len(input_ids)):
            token_id = input_ids[i].item()
            label_id = label_ids[i].item()
            token_text = processor.decode([token_id], skip_special_tokens=False)
            is_trained = "NO (masked)" if label_id == -100 else "YES"
            print(f"{token_id:<10} {repr(token_text):<30} {label_id:<10} {is_trained:<10}")

        # 8. What the model is trained to predict
        print("\n[7] TRAINING TARGET:")
        print("-" * 100)

        # Extract only the trainable tokens
        trainable_mask = label_ids != -100
        trainable_tokens = input_ids[trainable_mask]

        trainable_text = processor.decode(trainable_tokens, skip_special_tokens=True)
        print(f"Model is trained to output: {repr(trainable_text)}")
        print(f"Expected output: {repr(answer)}")

        print("\n" + "=" * 100)


def visualize_dpo_training_format(dataset_path: str, image_dir: str, processor, num_samples: int = 3):
    """Visualize how DPO data is formatted for training"""

    print("=" * 100)
    print("DPO TRAINING FORMAT VISUALIZATION")
    print("=" * 100)
    print()

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset[:num_samples]

    for idx, item in enumerate(dataset, 1):
        print("\n" + "=" * 100)
        print(f"SAMPLE {idx}")
        print("=" * 100)

        # 1. Original data
        print("\n[1] ORIGINAL DATA:")
        print("-" * 100)
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        image_name = item.get('image_name', '')

        print(f"Prompt: {prompt}")
        print(f"Chosen Response: {chosen}")
        print(f"Rejected Response: {rejected}")
        print(f"Image: {image_name}")

        # Load image
        image = None
        if image_name and Path(image_dir).exists():
            img_path = Path(image_dir) / image_name
            if img_path.exists():
                image = Image.open(img_path).convert('RGB')
                print(f"Image loaded: {img_path} (size: {image.size})")
            else:
                image = Image.new('RGB', (224, 224), color='white')
                print(f"Image not found, using blank image")
        else:
            image = Image.new('RGB', (224, 224), color='white')
            print(f"No image path, using blank image")

        # 2. Create messages for chosen response
        print("\n[2] FORMATTED MESSAGES (CHOSEN):")
        print("-" * 100)
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        chosen_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": chosen}]
            }
        ]

        rejected_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rejected}]
            }
        ]

        print("User Message:")
        print(json.dumps(user_message, indent=2))
        print("\nChosen Messages:")
        print(json.dumps(chosen_messages, indent=2))
        print("\nRejected Messages:")
        print(json.dumps(rejected_messages, indent=2))

        # 3. Apply chat template
        print("\n[3] CHAT TEMPLATE APPLIED:")
        print("-" * 100)
        prompt_text = processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        chosen_text = processor.apply_chat_template(chosen_messages, add_generation_prompt=False, tokenize=False)
        rejected_text = processor.apply_chat_template(rejected_messages, add_generation_prompt=False, tokenize=False)

        print("Prompt Text:")
        print(repr(prompt_text))
        print("\nChosen Text (prompt + chosen response):")
        print(repr(chosen_text))
        print("\nRejected Text (prompt + rejected response):")
        print(repr(rejected_text))

        # 4. DPO Training Info
        print("\n[4] DPO TRAINING:")
        print("-" * 100)
        print("DPO trains the model to:")
        print("  1. Increase log probability of CHOSEN response")
        print("  2. Decrease log probability of REJECTED response")
        print("  3. Maximize the margin: log P(chosen) - log P(rejected)")
        print()
        print("The model learns to prefer the chosen response over the rejected one")
        print("by optimizing a preference-based loss function.")

        # 5. Tokenize both
        print("\n[5] TOKENIZATION:")
        print("-" * 100)
        prompt_inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        chosen_inputs = processor(
            text=chosen_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        rejected_inputs = processor(
            text=rejected_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        prompt_length = prompt_inputs["input_ids"].shape[1]
        chosen_length = chosen_inputs["input_ids"].shape[1]
        rejected_length = rejected_inputs["input_ids"].shape[1]

        print(f"Prompt token length: {prompt_length}")
        print(f"Chosen sequence length: {chosen_length}")
        print(f"Rejected sequence length: {rejected_length}")
        print(f"Chosen response tokens: {chosen_length - prompt_length}")
        print(f"Rejected response tokens: {rejected_length - prompt_length}")

        # 6. Show what gets trained
        print("\n[6] TRAINING TARGET:")
        print("-" * 100)
        print("Model is trained to:")
        print(f"  PREFER (increase probability): {repr(chosen)}")
        print(f"  REJECT (decrease probability): {repr(rejected)}")

        print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Visualize training data format")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name: qcm_gemini, qcm_nova, qcm_procedure1, qcm_procedure2, dpo_gemini, dpo_nova")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to visualize (default: 3)")

    args = parser.parse_args()

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    print(f"Processor loaded: {BASE_MODEL}\n")

    # Get dataset paths
    base_path = Path(__file__).parent

    # Define dataset configurations (matching config.yaml)
    dataset_configs = {
        "qcm_gemini": {
            "type": "qcm",
            "dataset": "datasets/qcm/qcm_dataset_gemini.json",
            "image_dir": "datasets/images"
        },
        "qcm_nova": {
            "type": "qcm",
            "dataset": "datasets/qcm/qcm_dataset_nova_pro.json",
            "image_dir": "datasets/images"
        },
        "qcm_procedure1": {
            "type": "qcm",
            "dataset": "datasets/qcm/qcm_procedure1_claude_code.json",
            "image_dir": "datasets/procedureimages"
        },
        "qcm_procedure2": {
            "type": "qcm",
            "dataset": "datasets/qcm/qcm_procedure2_geminicli.json",
            "image_dir": "datasets/procedureimages"
        },
        "dpo_gemini": {
            "type": "dpo",
            "dataset": "datasets/dpo/dpo_dataset_gemini.json",
            "image_dir": "datasets/images"
        },
        "dpo_nova": {
            "type": "dpo",
            "dataset": "datasets/dpo/dpo_dataset_nova_pro.json",
            "image_dir": "datasets/images"
        }
    }

    if args.dataset not in dataset_configs:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Available datasets: {', '.join(dataset_configs.keys())}")
        return

    config = dataset_configs[args.dataset]
    dataset_path = base_path / config["dataset"]
    image_dir = base_path / config["image_dir"]

    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    if not image_dir.exists():
        print(f"Warning: Image directory not found at {image_dir}")
        print("Will use blank images for visualization")

    # Visualize based on dataset type
    if config["type"] == "qcm":
        visualize_qcm_training_format(
            str(dataset_path),
            str(image_dir),
            processor,
            args.num_samples
        )
    elif config["type"] == "dpo":
        visualize_dpo_training_format(
            str(dataset_path),
            str(image_dir),
            processor,
            args.num_samples
        )

    print("\n" + "=" * 100)
    print("VISUALIZATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
