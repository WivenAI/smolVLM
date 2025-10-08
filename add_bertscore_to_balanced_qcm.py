#!/usr/bin/env python3
"""
Script to add BERTScore results to balanced_qcm_all.json
"""

import json
from pathlib import Path


def add_bertscore_to_balanced_qcm(
    bertscore_file="dpo_bertscore_results.json",
    balanced_qcm_file="balanced_qcm_all.json",
    output_file="balanced_qcm_all.json"
):
    """Add BERTScore results to balanced_qcm_all.json"""

    # Load BERTScore results
    print(f"Loading BERTScore results from {bertscore_file}")
    with open(bertscore_file, 'r', encoding='utf-8') as f:
        bertscore_data = json.load(f)

    # Load balanced QCM data
    print(f"Loading balanced QCM data from {balanced_qcm_file}")
    with open(balanced_qcm_file, 'r', encoding='utf-8') as f:
        qcm_data = json.load(f)

    # Find the next available ID
    if qcm_data:
        next_id = max(item['id'] for item in qcm_data) + 1
    else:
        next_id = 1

    # Create a new entry for BERTScore benchmark
    bertscore_entry = {
        "id": next_id,
        "benchmark_name": "DPO Image Dataset - BERTScore Evaluation",
        "benchmark_type": "bertscore",
        "timestamp": bertscore_data["metadata"]["timestamp"],
        "dataset": {
            "path": bertscore_data["metadata"]["dataset_path"],
            "num_examples": bertscore_data["metadata"]["num_examples"]
        },
        "model": bertscore_data["metadata"]["model"],
        "overall_metrics": bertscore_data["overall_metrics"],
        "summary": {
            "precision_mean": bertscore_data["overall_metrics"]["precision"]["mean"],
            "recall_mean": bertscore_data["overall_metrics"]["recall"]["mean"],
            "f1_mean": bertscore_data["overall_metrics"]["f1"]["mean"],
            "precision_std": bertscore_data["overall_metrics"]["precision"]["std"],
            "recall_std": bertscore_data["overall_metrics"]["recall"]["std"],
            "f1_std": bertscore_data["overall_metrics"]["f1"]["std"]
        },
        "per_example_metrics": []
    }

    # Add per-example metrics (summary only, not full text to keep file size manageable)
    for example in bertscore_data["per_example_results"]:
        bertscore_entry["per_example_metrics"].append({
            "id": example["id"],
            "image_name": example["image_name"],
            "type": example["type"],
            "bertscore": example["bertscore"]
        })

    # Add to QCM data
    qcm_data.append(bertscore_entry)

    # Save updated QCM data
    print(f"Saving updated data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qcm_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*50)
    print("BERTSCORE RESULTS ADDED TO BALANCED QCM")
    print("="*50)
    print(f"New entry ID: {next_id}")
    print(f"Benchmark: DPO Image Dataset - BERTScore Evaluation")
    print(f"F1 Score: {bertscore_entry['summary']['f1_mean']:.4f} ± {bertscore_entry['summary']['f1_std']:.4f}")
    print(f"Precision: {bertscore_entry['summary']['precision_mean']:.4f} ± {bertscore_entry['summary']['precision_std']:.4f}")
    print(f"Recall: {bertscore_entry['summary']['recall_mean']:.4f} ± {bertscore_entry['summary']['recall_std']:.4f}")
    print("="*50)


def main():
    add_bertscore_to_balanced_qcm(
        bertscore_file="dpo_bertscore_results.json",
        balanced_qcm_file="balanced_qcm_all.json",
        output_file="balanced_qcm_all.json"
    )
    print("\nBERTScore results successfully added to balanced_qcm_all.json!")


if __name__ == "__main__":
    main()
