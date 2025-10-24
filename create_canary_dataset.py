#!/usr/bin/env python3
"""
Create a small canary dataset for testing if training actually works.
A canary dataset contains very simple, memorizable examples that a model
should easily learn to 100% accuracy if training is working correctly.

This helps detect:
- Training bugs (model should memorize small dataset)
- Data pipeline issues
- Overfitting capability (good for debugging)
"""

import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def create_simple_text_image(text: str, size=(400, 200), bg_color=(255, 255, 255)):
    """Create a simple image with text on it"""
    image = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(image)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
    except:
        font = ImageFont.load_default()

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)

    # Draw text
    draw.text(position, text, fill=(0, 0, 0), font=font)

    return image


def create_canary_qcm_dataset(output_dir: Path, num_samples: int = 5):
    """Create canary QCM dataset with simple, memorizable examples"""

    canary_data = []

    # Very simple, distinctive examples
    canaries = [
        {
            "text": "APPLE",
            "question": "What fruit is shown in this image?",
            "options": {"A": "Apple", "B": "Banana", "C": "Orange", "D": "Grape"},
            "correct": "A",
            "explanation": "The image clearly shows the word APPLE"
        },
        {
            "text": "BLUE",
            "question": "What color is written in this image?",
            "options": {"A": "Red", "B": "Blue", "C": "Green", "D": "Yellow"},
            "correct": "B",
            "explanation": "The text says BLUE"
        },
        {
            "text": "CAT",
            "question": "What animal is mentioned in this image?",
            "options": {"A": "Dog", "B": "Cat", "C": "Bird", "D": "Fish"},
            "correct": "B",
            "explanation": "The word CAT is displayed"
        },
        {
            "text": "THREE",
            "question": "What number word is shown?",
            "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
            "correct": "C",
            "explanation": "The image shows THREE"
        },
        {
            "text": "SQUARE",
            "question": "What shape is named in this image?",
            "options": {"A": "Circle", "B": "Square", "C": "Triangle", "D": "Rectangle"},
            "correct": "B",
            "explanation": "SQUARE is written in the image"
        },
    ]

    # Use only requested number of samples
    canaries = canaries[:num_samples]

    for idx, canary in enumerate(canaries):
        # Create image
        image = create_simple_text_image(canary["text"])
        image_filename = f"canary_{idx:03d}.png"
        image_path = output_dir / image_filename
        image.save(image_path)

        # Create dataset entry
        entry = {
            "image_name": image_filename,
            "qcm": {
                "question": canary["question"],
                "options": canary["options"],
                "correct_answer": canary["correct"],
                "explanation": canary["explanation"]
            }
        }

        canary_data.append(entry)

    return canary_data


def create_canary_dpo_dataset(output_dir: Path, num_samples: int = 5):
    """Create canary DPO dataset with clear chosen/rejected pairs"""

    canary_data = []

    # Very simple examples with obvious correct/incorrect responses
    canaries = [
        {
            "text": "DOG",
            "prompt": "What animal is shown in this image?",
            "chosen": "The image shows the word DOG, which refers to a common pet animal.",
            "rejected": "I cannot see any animal in this image. The image is blank."
        },
        {
            "text": "RED",
            "prompt": "What color is written in this image?",
            "chosen": "The word RED is clearly visible in the image.",
            "rejected": "There is no text visible in this image."
        },
        {
            "text": "FIVE",
            "prompt": "What number word appears in the image?",
            "chosen": "The image displays the word FIVE.",
            "rejected": "The image shows the number 10, not a word."
        },
        {
            "text": "CIRCLE",
            "prompt": "What shape name is shown?",
            "chosen": "The text CIRCLE is shown in the image.",
            "rejected": "I see a square shape drawn in the image."
        },
        {
            "text": "HELLO",
            "prompt": "What greeting is displayed?",
            "chosen": "The word HELLO is displayed as a greeting.",
            "rejected": "The image says GOODBYE."
        },
    ]

    # Use only requested number of samples
    canaries = canaries[:num_samples]

    for idx, canary in enumerate(canaries):
        # Create image
        image = create_simple_text_image(canary["text"])
        image_filename = f"canary_dpo_{idx:03d}.png"
        image_path = output_dir / image_filename
        image.save(image_path)

        # Create dataset entry
        entry = {
            "image_name": image_filename,
            "prompt": canary["prompt"],
            "chosen": canary["chosen"],
            "rejected": canary["rejected"]
        }

        canary_data.append(entry)

    return canary_data


def main():
    parser = argparse.ArgumentParser(
        description="Create canary datasets for testing if training works"
    )
    parser.add_argument("--output-dir", type=str, default="canary_dataset",
                       help="Output directory for canary dataset")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of canary samples to create (max 5)")
    parser.add_argument("--type", choices=["qcm", "dpo", "both"], default="both",
                       help="Type of canary dataset to create")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("CREATING CANARY DATASET")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Dataset type: {args.type}\n")

    # Limit to max 5 samples
    num_samples = min(args.num_samples, 5)

    if args.type in ["qcm", "both"]:
        print("Creating QCM canary dataset...")
        qcm_data = create_canary_qcm_dataset(output_dir, num_samples)
        qcm_path = output_dir / "canary_qcm.json"
        with open(qcm_path, 'w') as f:
            json.dump(qcm_data, f, indent=2)
        print(f"✅ Created {len(qcm_data)} QCM canary examples: {qcm_path}")

    if args.type in ["dpo", "both"]:
        print("\nCreating DPO canary dataset...")
        dpo_data = create_canary_dpo_dataset(output_dir, num_samples)
        dpo_path = output_dir / "canary_dpo.json"
        with open(dpo_path, 'w') as f:
            json.dump(dpo_data, f, indent=2)
        print(f"✅ Created {len(dpo_data)} DPO canary examples: {dpo_path}")

    print("\n" + "="*80)
    print("CANARY DATASET CREATED SUCCESSFULLY")
    print("="*80)
    print("\nUsage:")
    print("  1. Train on canary dataset")
    print("  2. Evaluate on same canary dataset")
    print("  3. Model should achieve ~100% accuracy")
    print("  4. If not, there's a bug in training code")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
