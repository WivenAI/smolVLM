#!/usr/bin/env python3
"""
Complete proof that OCRBench training works:
1. Split dataset into train (700) and test (300) with fixed seed
2. Evaluate BASE model on test set
3. Train model on train set
4. Evaluate TRAINED model on same test set
5. Compare results to prove training improves performance
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
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import argparse


def load_and_split_dataset(benchmark_name: str, train_size: int = 700, test_size: int = 300, seed: int = 42):
    """Load dataset and split into train/test with fixed seed"""
    print(f"\nLoading {benchmark_name} dataset...")

    if benchmark_name == "ocrbench":
        try:
            dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)
        except:
            dataset = load_dataset("lmms-lab/OCRBench-v2", split="test", trust_remote_code=True)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    print(f"Total samples: {len(dataset)}")

    # Split with fixed seed for reproducibility
    import random
    random.seed(seed)
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    train_indices = sorted(all_indices[:train_size])
    test_indices = sorted(all_indices[train_size:train_size + test_size])

    train_dataset = dataset.select(train_indices)
    test_dataset = dataset.select(test_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def evaluate_model(model_path: str, dataset, processor, device, description: str):
    """Evaluate a model on given dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {description}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load model
    if model_path == "base":
        model_path = "HuggingFaceTB/SmolVLM-500M-Instruct"

    print(f"Loading model from {model_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    results = []
    correct = 0
    total = 0

    print(f"Evaluating on {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]

        # Extract image
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            continue

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

        # Extract ground truth
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

        # Ensure ground_truth is a string
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

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

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

        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # Extract assistant response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        elif "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
            response = response.lstrip(":").strip()
        else:
            response = generated_text.strip()

        # Check if correct
        is_correct = ground_truth.lower().strip() in response.lower().strip() or \
                     response.lower().strip() in ground_truth.lower().strip()

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "correct": is_correct
        })

        # Show first few examples
        if idx < 3:
            print(f"\n--- Example {idx + 1} ---")
            print(f"Question: {question}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Response: {response}")
            print(f"Correct: {is_correct}")

    accuracy = (correct / total * 100) if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"{description} RESULTS")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results, accuracy


class OCRBenchDataset(torch.utils.data.Dataset):
    """Dataset for OCRBench training"""

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract image
        if 'image' in item:
            image = item['image']
        elif 'img' in item:
            image = item['img']
        else:
            raise ValueError("No image field")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to avoid memory issues
        max_size = 1024
        if image.size[0] > max_size or image.size[1] > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

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

        # Extract answer
        if 'answers' in item:
            answers = item['answers']
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0]
            else:
                answer = str(answers)
        elif 'answer' in item:
            answer = item['answer']
        elif 'label' in item:
            answer = str(item['label'])
        else:
            answer = "Unknown"

        # Ensure answer is string
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else "Unknown"
        answer = str(answer)

        # Apply chat template
        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        full_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        prompt_text = self.processor.apply_chat_template(user_message, add_generation_prompt=True, tokenize=False)
        full_text = self.processor.apply_chat_template(full_messages, add_generation_prompt=False, tokenize=False)

        # Process with image
        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Create labels with masking
        prompt_length = prompt_inputs["input_ids"].shape[1]
        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100

        return {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "pixel_values": full_inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0)
        }


def train_model(train_dataset, processor, output_dir: str, num_epochs: int = 1):
    """Train model on training dataset"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL")
    print(f"{'='*60}")

    # Load base model with quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create dataset
    dataset = OCRBenchDataset(train_dataset, processor)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="adamw_8bit",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nStarting training on {len(dataset)} samples...")
    trainer.train()

    print(f"\nSaving model to {output_dir}...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    print(f"\nTraining completed!")

    # Clean up
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Prove OCRBench training works")
    parser.add_argument("--benchmark", type=str, default="ocrbench", help="Benchmark name")
    parser.add_argument("--train-size", type=int, default=700, help="Training samples")
    parser.add_argument("--test-size", type=int, default=300, help="Test samples")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--output-dir", type=str, default="./proof_trained_model", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", trust_remote_code=True)

    # Step 1: Load and split dataset
    train_dataset, test_dataset = load_and_split_dataset(
        args.benchmark,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )

    # Step 2: Evaluate BASE model on test set
    base_results, base_accuracy = evaluate_model(
        "base",
        test_dataset,
        processor,
        device,
        "BASE MODEL (before training)"
    )

    # Step 3: Train model on train set
    train_model(train_dataset, processor, args.output_dir, args.num_epochs)

    # Step 4: Evaluate TRAINED model on same test set
    trained_results, trained_accuracy = evaluate_model(
        args.output_dir,
        test_dataset,
        processor,
        device,
        "TRAINED MODEL (after training)"
    )

    # Step 5: Compare results
    print(f"\n{'#'*60}")
    print(f"FINAL COMPARISON - PROOF THAT TRAINING WORKS")
    print(f"{'#'*60}")
    print(f"Dataset: {args.benchmark}")
    print(f"Train samples: {args.train_size}")
    print(f"Test samples: {args.test_size}")
    print(f"Seed: {args.seed}")
    print(f"\nBASE MODEL accuracy:    {base_accuracy:.2f}%")
    print(f"TRAINED MODEL accuracy: {trained_accuracy:.2f}%")
    print(f"\nImprovement: {trained_accuracy - base_accuracy:+.2f}%")

    if trained_accuracy > base_accuracy:
        print(f"\n✅ SUCCESS! Training IMPROVES the model!")
        print(f"   Trained model is {trained_accuracy - base_accuracy:.2f}% better")
    elif trained_accuracy == base_accuracy:
        print(f"\n⚠️  WARNING: No improvement detected")
    else:
        print(f"\n❌ FAILURE: Training made model WORSE!")
    print(f"{'#'*60}\n")

    # Save comparison results
    comparison = {
        "benchmark": args.benchmark,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "seed": args.seed,
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "improvement": trained_accuracy - base_accuracy,
        "base_results": base_results,
        "trained_results": trained_results
    }

    with open("proof_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Detailed results saved to: proof_comparison.json")


if __name__ == "__main__":
    main()
