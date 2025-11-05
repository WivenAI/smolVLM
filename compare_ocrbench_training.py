#!/usr/bin/env python3
"""
Compare base model vs OCRBench-trained model performance
"""

import json

def calculate_accuracy(results):
    """Calculate accuracy from results array"""
    if not results:
        return 0.0

    correct = 0
    total = len(results)

    for item in results:
        response = item.get('response', '').lower()
        ground_truth = item.get('ground_truth', [])

        # Check if response matches any ground truth
        if isinstance(ground_truth, list):
            if any(gt.lower() in response for gt in ground_truth):
                correct += 1
        elif isinstance(ground_truth, str):
            if ground_truth.lower() in response:
                correct += 1

    return correct / total if total > 0 else 0.0

# Load files
print('='*80)
print('OCRBench SFT Training Impact (Oct 31, 2025)')
print('='*80)

with open('systematic_results/base_model_20251031_092613.json', 'r') as f:
    base_data = json.load(f)

with open('systematic_results/trained_on_ocrbench_20251031_110759.json', 'r') as f:
    ocrbench_data = json.load(f)

print(f'\n{"Dataset":25s} Base Model  After Training  Difference')
print('-'*70)

results = {}

for dataset in sorted(set(list(base_data.keys()) + list(ocrbench_data.keys()))):
    if dataset in base_data and dataset in ocrbench_data:
        base_acc = calculate_accuracy(base_data[dataset])
        trained_acc = calculate_accuracy(ocrbench_data[dataset])
        diff = trained_acc - base_acc
        symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "=")

        results[dataset] = {
            'base': base_acc,
            'trained': trained_acc,
            'diff': diff
        }

        print(f'{dataset:25s} {base_acc:7.2%}      {trained_acc:7.2%}       {diff:+.2%} {symbol}')

# Key findings
print(f'\n📊 KEY FINDINGS:')
if 'ocrbench' in results:
    ocr = results['ocrbench']
    print(f'\n1. OCRBench Performance:')
    if ocr['diff'] > 0:
        print(f'   ✅ Training on OCRBench IMPROVES performance on OCRBench')
        print(f'      {ocr["base"]:.2%} → {ocr["trained"]:.2%} (+{ocr["diff"]:.2%})')
    elif ocr['diff'] == 0:
        print(f'   ⚠️  Training on OCRBench has NO EFFECT')
        print(f'      {ocr["base"]:.2%} → {ocr["trained"]:.2%} (no change)')
    else:
        print(f'   ❌ Training on OCRBench DECREASES performance')
        print(f'      {ocr["base"]:.2%} → {ocr["trained"]:.2%} ({ocr["diff"]:.2%})')

print(f'\n2. Transfer Learning Effects:')
for dataset in results:
    if dataset != 'ocrbench':
        r = results[dataset]
        if abs(r['diff']) > 0.01:  # Significant change
            direction = "improved" if r['diff'] > 0 else "decreased"
            print(f'   • {dataset}: {direction} by {abs(r["diff"]):.2%}')
