#!/usr/bin/env python3
"""
Script to add DPO Log Probability results to balanced_qcm_all.json
"""

import json
from pathlib import Path


def add_dpo_logprob_to_balanced_qcm(
    logprob_file="dpo_logprob_results.json",
    balanced_qcm_file="balanced_qcm_all.json",
    output_file="balanced_qcm_all.json"
):
    """Add DPO Log Probability results to balanced_qcm_all.json"""

    # Load DPO Log Probability results
    print(f"Loading DPO Log Probability results from {logprob_file}")
    with open(logprob_file, 'r', encoding='utf-8') as f:
        logprob_data = json.load(f)

    # Load balanced QCM data
    print(f"Loading balanced QCM data from {balanced_qcm_file}")
    with open(balanced_qcm_file, 'r', encoding='utf-8') as f:
        qcm_data = json.load(f)

    # Find the next available ID
    if qcm_data:
        next_id = max(item['id'] for item in qcm_data) + 1
    else:
        next_id = 1

    # Create a new entry for DPO Log Probability benchmark
    logprob_entry = {
        "id": next_id,
        "benchmark_name": "DPO Image Dataset - Log Probability Evaluation",
        "benchmark_type": "dpo_logprob",
        "timestamp": logprob_data["metadata"]["timestamp"],
        "dataset": {
            "path": logprob_data["metadata"]["dataset_path"],
            "num_examples": logprob_data["metadata"]["num_examples"]
        },
        "model": logprob_data["metadata"]["model"],
        "overall_metrics": logprob_data["overall_metrics"],
        "summary": {
            "preference_accuracy": logprob_data["overall_metrics"]["preference_accuracy"],
            "num_correct_preferences": logprob_data["overall_metrics"]["num_correct_preferences"],
            "margin_mean": logprob_data["overall_metrics"]["margin"]["mean"],
            "chosen_avg_logprob_mean": logprob_data["overall_metrics"]["chosen_avg_logprob"]["mean"],
            "rejected_avg_logprob_mean": logprob_data["overall_metrics"]["rejected_avg_logprob"]["mean"],
            "chosen_perplexity_mean": logprob_data["overall_metrics"]["chosen_perplexity"]["mean"],
            "rejected_perplexity_mean": logprob_data["overall_metrics"]["rejected_perplexity"]["mean"]
        },
        "per_example_metrics": []
    }

    # Add per-example metrics (summary only)
    for example in logprob_data["per_example_results"]:
        logprob_entry["per_example_metrics"].append({
            "id": example["id"],
            "image_name": example["image_name"],
            "type": example["type"],
            "margin": example["margin"],
            "preference_correct": example["preference_correct"],
            "chosen_avg_logprob": example["chosen_metrics"]["avg_logprob"],
            "rejected_avg_logprob": example["rejected_metrics"]["avg_logprob"]
        })

    # Add to QCM data
    qcm_data.append(logprob_entry)

    # Save updated QCM data
    print(f"Saving updated data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qcm_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("DPO LOG PROBABILITY RESULTS ADDED TO BALANCED QCM")
    print("="*80)
    print(f"New entry ID: {next_id}")
    print(f"Benchmark: DPO Image Dataset - Log Probability Evaluation")
    print(f"Preference Accuracy: {logprob_entry['summary']['preference_accuracy']:.2%}")
    print(f"Margin (mean): {logprob_entry['summary']['margin_mean']:.4f}")
    print(f"Chosen Log Prob: {logprob_entry['summary']['chosen_avg_logprob_mean']:.4f}")
    print(f"Rejected Log Prob: {logprob_entry['summary']['rejected_avg_logprob_mean']:.4f}")
    print("="*80)


def main():
    add_dpo_logprob_to_balanced_qcm(
        logprob_file="dpo_logprob_results.json",
        balanced_qcm_file="balanced_qcm_all.json",
        output_file="balanced_qcm_all.json"
    )
    print("\nDPO Log Probability results successfully added to balanced_qcm_all.json!")


if __name__ == "__main__":
    main()
