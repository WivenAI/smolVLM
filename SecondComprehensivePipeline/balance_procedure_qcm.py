#!/usr/bin/env python3
"""
Balance procedure QCM datasets to ensure no answer (A/B/C/D) exceeds 28%.

This script shuffles the options for each question to redistribute correct answers.

Usage:
    python balance_procedure_qcm.py datasets/qcm/qcm_procedure1_claude_code.json
    python balance_procedure_qcm.py datasets/qcm/qcm_procedure2_geminicli.json
    python balance_procedure_qcm.py --all  # Balance all procedure datasets
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter


def balance_qcm_dataset(input_path: str, output_path: str = None, seed: int = 42):
    """Balance a QCM dataset by shuffling options"""

    if output_path is None:
        output_path = input_path

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load dataset
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing: {input_path.name}")
    print(f"  Total questions: {len(data)}")

    # Count original distribution
    original_counts = Counter()
    for item in data:
        ans = item.get("correct_answer") or item.get("qcm", {}).get("correct_answer", "")
        original_counts[ans.strip().upper()] += 1

    print(f"  Original distribution:")
    for letter in sorted(original_counts.keys()):
        pct = (original_counts[letter] / len(data) * 100)
        print(f"    {letter}: {original_counts[letter]:3d} ({pct:5.1f}%)")

    # Create balanced distribution of correct answers
    random.seed(seed)
    letters = ['A', 'B', 'C', 'D']
    total_questions = len(data)

    # Target counts (as even as possible)
    base_count = total_questions // 4
    remainder = total_questions % 4

    # Distribute remainder across letters
    target_counts = {letter: base_count for letter in letters}
    for i in range(remainder):
        target_counts[letters[i]] += 1

    print(f"  Target distribution:")
    for letter in sorted(target_counts.keys()):
        pct = (target_counts[letter] / total_questions * 100)
        print(f"    {letter}: {target_counts[letter]:3d} ({pct:5.1f}%)")

    # Create list of target answer letters
    target_answers = []
    for letter, count in target_counts.items():
        target_answers.extend([letter] * count)

    # Shuffle the target answers
    random.shuffle(target_answers)

    # Assign target answers and shuffle options
    for item, target_letter in zip(data, target_answers):
        # Handle nested structure
        if "qcm" in item:
            qcm = item["qcm"]
        else:
            qcm = item

        # Get current correct answer and its text
        correct_letter = qcm["correct_answer"].strip().upper()
        correct_text = qcm["options"][correct_letter]

        # Get all option texts
        option_texts = [qcm["options"][letter] for letter in letters]

        # Shuffle so correct text ends up at target letter
        # Remove correct text from list
        option_texts_without_correct = [t for t in option_texts if t != correct_text]

        # Shuffle the other options
        random.shuffle(option_texts_without_correct)

        # Build new options with correct answer at target position
        new_option_texts = []
        other_idx = 0
        for letter in letters:
            if letter == target_letter:
                new_option_texts.append(correct_text)
            else:
                new_option_texts.append(option_texts_without_correct[other_idx])
                other_idx += 1

        # Create new options mapping
        new_options = {letter: text for letter, text in zip(letters, new_option_texts)}

        # Update the item
        qcm["options"] = new_options
        qcm["correct_answer"] = target_letter

    # Count new distribution
    new_counts = Counter()
    for item in data:
        ans = item.get("correct_answer") or item.get("qcm", {}).get("correct_answer", "")
        new_counts[ans.strip().upper()] += 1

    print(f"  New distribution:")
    for letter in sorted(new_counts.keys()):
        pct = (new_counts[letter] / len(data) * 100)
        status = "✓" if pct <= 28 else "✗"
        print(f"    {status} {letter}: {new_counts[letter]:3d} ({pct:5.1f}%)")

    # Save balanced dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {output_path}")

    # Check if balanced
    max_pct = max(new_counts[letter] / len(data) * 100 for letter in letters)
    if max_pct <= 28:
        print(f"  ✓ Dataset is now balanced!")
    else:
        print(f"  ⚠ Still imbalanced (max {max_pct:.1f}%), run again with different seed")

    return max_pct <= 28


def main():
    parser = argparse.ArgumentParser(description="Balance procedure QCM datasets")
    parser.add_argument("dataset", nargs="?", help="Path to dataset file")
    parser.add_argument("--all", action="store_true", help="Balance all procedure datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Output file (default: overwrite input)")

    args = parser.parse_args()

    if args.all:
        # Balance all procedure datasets
        base_path = Path(__file__).parent
        datasets = [
            "datasets/qcm/qcm_procedure1_claude_code.json",
            "datasets/qcm/qcm_procedure2_geminicli.json"
        ]

        print("="*80)
        print("Balancing all procedure datasets")
        print("="*80)

        for ds in datasets:
            ds_path = base_path / ds
            if ds_path.exists():
                print()
                balance_qcm_dataset(str(ds_path), seed=args.seed)
            else:
                print(f"⚠ Not found: {ds}")

        print("\n" + "="*80)
        print("Done! Run 'python test_datasets.py' to verify")
        print("="*80)

    elif args.dataset:
        balance_qcm_dataset(args.dataset, args.output, args.seed)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
