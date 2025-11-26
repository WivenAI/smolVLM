# ERP DPO Evaluation Results

Generated: 2025-11-10 12:00:35

All models evaluated on ERP DPO dataset (1833-1868 samples)

## Summary Table

| Model Configuration | DPO LogProb (margin) | Preference Acc. (%) | BERTScore F1 (%) | Dataset Size |
|---------------------|----------------------|---------------------|------------------|--------------|
| Base Model | 0.7587 | 97.27 | 61.87 | 1833 |
| Trained: Chartqa | 0.4684 | 88.54 | 58.30 | 1833 |
| Trained: Ocrbench | 0.4883 | 88.49 | 62.30 | 1833 |
| Trained: Docvqa | 0.4511 | 87.07 | 59.57 | 1833 |
| Trained: Erp Qcm | 0.6076 | 94.11 | 62.07 | 1833 |
| Trained: Erp Dpo | 1.0636 | 95.34 | 63.10 | 1868 |
| Trained On Erp Dpo Sft | N/A | N/A | N/A | N/A |
| Trained: Erp Qcm+Dpo | 1.0638 | 95.34 | 63.07 | 1868 |
| Trained On Erp Qcm+Dpo Sft | N/A | N/A | N/A | N/A |

## Notes

- **DPO LogProb (margin)**: Higher is better. Measures how much more likely the model is to generate chosen vs rejected responses.
- **Preference Accuracy**: Higher is better. Percentage of times the model assigns higher probability to chosen responses.
- **BERTScore F1**: Higher is better. Semantic similarity between generated and reference responses.

## Key Findings

1. **Best performers**: ERP DPO and ERP QCM+DPO (margin ~1.06, BERTScore F1 ~63%)
2. **Base model**: Surprisingly strong performance (margin 0.76, pref. acc. 97.27%)
3. **Domain transfer**: Models trained on other datasets (ChartQA, OCRBench, DocVQA) show degraded performance on ERP DPO
4. **Missing**: DPO-SFT variants need to be trained/evaluated

## Source Files

Results are loaded from the most recent evaluation files in `systematic_results/`:

- **base_model**: `base_model_erp_dpo_20251107_185418.json`
- **trained_on_chartqa**: `trained_on_chartqa_erp_dpo_20251108_021352.json`
- **trained_on_ocrbench**: `trained_on_ocrbench_erp_dpo_20251107_231504.json`
- **trained_on_docvqa**: `trained_on_docvqa_erp_dpo_20251107_203639.json`
- **trained_on_erp_qcm**: `trained_on_erp_qcm_erp_dpo_20251108_152041.json`
- **trained_on_erp_dpo**: `trained_on_erp_dpo_erp_dpo_20251109_031505.json`
- **trained_on_erp_qcm+dpo**: `trained_on_erp_qcm+dpo_erp_dpo_20251110_035239.json`