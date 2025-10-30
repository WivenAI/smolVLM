#!/usr/bin/env python3
"""
Analyze all existing training results and create comprehensive comparison
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_all_results():
    """Load all systematic comparison results"""
    results_dir = Path("systematic_results")

    all_results = defaultdict(list)

    for json_file in sorted(results_dir.glob("systematic_comparison_*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            timestamp = json_file.stem.replace("systematic_comparison_", "")

            for model_name, model_data in data.items():
                metrics = model_data.get('metrics', {})
                if metrics:
                    result = {
                        'timestamp': timestamp,
                        'model': model_name,
                        'file': json_file.name
                    }

                    for bench, bench_metrics in metrics.items():
                        result[f'{bench}_acc'] = bench_metrics.get('accuracy', 0)
                        result[f'{bench}_samples'] = bench_metrics.get('num_samples', 0)

                    # Calculate average
                    accs = [m.get('accuracy', 0) for m in metrics.values()]
                    result['average_accuracy'] = sum(accs) / len(accs) if accs else 0

                    all_results[model_name].append(result)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return all_results


def find_best_results(all_results):
    """Find the best result for each model"""
    best_results = {}

    for model_name, results in all_results.items():
        if results:
            # Sort by average accuracy (descending)
            sorted_results = sorted(results, key=lambda x: x['average_accuracy'], reverse=True)
            best_results[model_name] = sorted_results[0]

    return best_results


def main():
    print("="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)

    # Load all results
    print("\n1. Loading all results...")
    all_results = load_all_results()

    print(f"   Found {len(all_results)} unique models")
    for model_name in sorted(all_results.keys()):
        print(f"      - {model_name}: {len(all_results[model_name])} result(s)")

    # Find best results
    print("\n2. Finding best result for each model...")
    best_results = find_best_results(all_results)

    # Create comparison dataframe
    comparison_data = []
    for model_name, result in best_results.items():
        comparison_data.append(result)

    df = pd.DataFrame(comparison_data)

    # Sort by average accuracy
    df = df.sort_values('average_accuracy', ascending=False)

    # Print results
    print("\n" + "="*80)
    print("BEST RESULTS FOR EACH MODEL")
    print("="*80 + "\n")

    # Select columns to display
    display_cols = ['model', 'average_accuracy']
    bench_cols = [col for col in df.columns if col.endswith('_acc') and col != 'average_accuracy']
    display_cols.extend(sorted(bench_cols))

    # Format and print
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)

    print(df[display_cols].to_string(index=False))

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Check if we have base_model
    if 'base_model' in best_results:
        base_acc = best_results['base_model']['average_accuracy']
        print(f"\n📊 BASE MODEL PERFORMANCE:")
        print(f"   Average Accuracy: {base_acc:.2f}%")

        base_result = best_results['base_model']
        for bench_col in sorted(bench_cols):
            if bench_col in base_result:
                bench_name = bench_col.replace('_acc', '')
                print(f"   {bench_name}: {base_result[bench_col]:.2f}%")

        # Compare trained models
        print(f"\n📈 IMPROVEMENTS OVER BASELINE:")
        improvements = []
        for model_name, result in best_results.items():
            if model_name != 'base_model':
                improvement = result['average_accuracy'] - base_acc
                improvements.append({
                    'model': model_name,
                    'avg_acc': result['average_accuracy'],
                    'improvement': improvement
                })

        improvements.sort(key=lambda x: x['improvement'], reverse=True)

        for item in improvements:
            symbol = "📈" if item['improvement'] > 0 else "📉" if item['improvement'] < 0 else "➡️"
            print(f"   {symbol} {item['model']:40s}: {item['avg_acc']:.2f}% ({item['improvement']:+.2f}%)")

    # OCRBench specific analysis
    print(f"\n🔍 OCRBENCH SPECIFIC ANALYSIS:")

    if 'base_model' in best_results and 'ocrbench_acc' in best_results['base_model']:
        base_ocr = best_results['base_model']['ocrbench_acc']
        print(f"   Base model OCRBench: {base_ocr:.2f}%")

        if 'trained_on_ocrbench' in best_results and 'ocrbench_acc' in best_results['trained_on_ocrbench']:
            trained_ocr = best_results['trained_on_ocrbench']['ocrbench_acc']
            diff = trained_ocr - base_ocr
            symbol = "✅" if diff > 0 else "❌" if diff < 0 else "➡️"

            print(f"   Trained on OCRBench: {trained_ocr:.2f}%")
            print(f"   {symbol} Difference: {diff:+.2f}%")

            if diff > 0:
                print(f"   ✅ Training on OCRBench IMPROVED OCRBench performance!")
            elif diff < 0:
                print(f"   ⚠️  Training on OCRBench DECREASED OCRBench performance")
            else:
                print(f"   ➡️  Training on OCRBench had NO EFFECT on OCRBench performance")
        else:
            print("   ⚠️  No trained_on_ocrbench results found")
    else:
        print("   ⚠️  No base_model OCRBench results found")

    # DocVQA analysis
    print(f"\n📄 DOCVQA SPECIFIC ANALYSIS:")

    if 'base_model' in best_results and 'docvqa_acc' in best_results['base_model']:
        base_doc = best_results['base_model']['docvqa_acc']
        print(f"   Base model DocVQA: {base_doc:.2f}%")

        if 'trained_on_docvqa' in best_results and 'docvqa_acc' in best_results['trained_on_docvqa']:
            trained_doc = best_results['trained_on_docvqa']['docvqa_acc']
            diff = trained_doc - base_doc
            symbol = "✅" if diff > 0 else "❌" if diff < 0 else "➡️"

            print(f"   Trained on DocVQA: {trained_doc:.2f}%")
            print(f"   {symbol} Difference: {diff:+.2f}%")

    # ChartQA analysis
    print(f"\n📊 CHARTQA SPECIFIC ANALYSIS:")

    if 'base_model' in best_results and 'chartqa_acc' in best_results['base_model']:
        base_chart = best_results['base_model']['chartqa_acc']
        print(f"   Base model ChartQA: {base_chart:.2f}%")

        if 'trained_on_chartqa' in best_results and 'chartqa_acc' in best_results['trained_on_chartqa']:
            trained_chart = best_results['trained_on_chartqa']['chartqa_acc']
            diff = trained_chart - base_chart
            symbol = "✅" if diff > 0 else "❌" if diff < 0 else "➡️"

            print(f"   Trained on ChartQA: {trained_chart:.2f}%")
            print(f"   {symbol} Difference: {diff:+.2f}%")

    # ERP analysis
    print(f"\n🏢 ERP TRAINING ANALYSIS:")

    erp_models = [m for m in best_results.keys() if 'erp' in m.lower()]

    if erp_models:
        print(f"   Found {len(erp_models)} ERP-trained model(s):")
        for erp_model in sorted(erp_models):
            avg_acc = best_results[erp_model]['average_accuracy']
            print(f"      - {erp_model}: {avg_acc:.2f}%")
    else:
        print("   ⚠️  No ERP-trained models found")

    # Save summary
    output_file = Path("RESULTS_SUMMARY.csv")
    df[display_cols].to_csv(output_file, index=False)
    print(f"\n💾 Summary saved to: {output_file}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
