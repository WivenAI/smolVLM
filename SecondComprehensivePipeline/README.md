# SecondComprehensivePipeline

A modular training and evaluation pipeline for fine-tuning vision-language models (VLMs) using multiple strategies including SFT, DPO, and full fine-tuning. Built around SmolVLM2 and Qwen2-VL architectures.

## Overview

This pipeline supports a comprehensive set of training strategies for fine-tuning VLMs through:

- **SFT (Supervised Fine-Tuning)** on benchmark datasets (DocVQA, OCRBench, ChartQA)
- **DPO (Direct Preference Optimization)** using chosen/rejected response pairs
- **Full Fine-Tuning** on QCM (Question-Choice-Matching) datasets
- **Multi-stage training** (e.g., QCM then DPO, DPO then QCM)
- **Combined dataset training** merging multiple data sources

## Project Structure

```
SecondComprehensivePipeline/
├── pipeline.py                  # Main pipeline orchestrator
├── config/
│   ├── conf.yaml                # Main configuration file
│   ├── individual/              # Per-strategy config files
│   ├── overfitindividual/       # Overfit debug configs
│   └── setup.py                 # Config utilities
├── trainers/
│   ├── trainer_sft.py           # SFT trainer
│   ├── trainer_dpo.py           # DPO trainer
│   ├── trainer_full_finetune.py # Full fine-tuning trainer
│   ├── model_utils.py           # Model loading utilities
│   ├── image_utils.py           # Image processing utilities
│   └── callbacks/               # Training callbacks (e.g., epoch evaluation)
├── dataloader/
│   ├── base_dataset.py          # Base dataset class
│   ├── benchmark_dataset.py     # Benchmark dataset loaders (DocVQA, OCRBench, ChartQA)
│   ├── qcm_dataset.py           # QCM dataset loader
│   ├── dpo_dataset.py           # DPO dataset loader
│   └── data_collators.py        # Data collation utilities
├── evaluators/
│   ├── base_evaluator.py        # Base evaluator
│   ├── evaluator_qcm.py         # QCM accuracy evaluator
│   ├── evaluator_docvqa.py      # DocVQA evaluator
│   ├── evaluator_chartqa.py     # ChartQA evaluator
│   ├── evaluator_ocr.py         # OCRBench evaluator
│   ├── evaluator_bertscore.py   # BERTScore evaluator
│   ├── evaluator_rouge.py       # ROUGE evaluator
│   ├── evaluator_logprob.py     # Log-probability evaluator
│   └── evaluator_all.py         # Combined evaluator runner
├── datasets/                    # Dataset files (JSON + images)
├── tests/                       # Unit and integration tests
├── utils/
│   └── dual_logger.py           # Logging utilities
├── job.sh                       # SLURM job script
├── submit_all_jobs.sh           # Batch job submission
└── requirements.txt             # Python dependencies
```

## Usage

### Full pipeline

```bash
python pipeline.py                     # Run training + evaluation
python pipeline.py --eval-only         # Evaluation only
python pipeline.py --config config/individual/conf_sft_docvqa.yaml  # Custom config
```

### Individual training runs (SLURM)

```bash
# Submit all jobs
bash submit_all_jobs.sh

# Submit specific model jobs
bash submit_smolvlm2_2B_jobs.sh
bash submit_qwen2_vl_2B_jobs.sh
```

### Debug mode

Set `debug_mode: true` in the config to run with minimal samples for quick iteration.

## Configuration

Training strategies are defined in `config/conf.yaml`. Each strategy specifies:

- **name**: Unique identifier
- **type**: Training type (`sft_benchmark`, `dpo`, `full_ft_qcm`, `dpo_qcm`, etc.)
- **dataset**: Path to training data
- **image_dir**: Path to associated images
- **enabled**: Toggle on/off

Individual configs in `config/individual/` allow running single strategies as standalone SLURM jobs.

## Evaluation

Models are evaluated on:

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| DocVQA | `nielsr/docvqa_1200_examples` | ANLS |
| OCRBench | `echo840/OCRBench` | Accuracy |
| ChartQA | `HuggingFaceM4/ChartQA` | Relaxed accuracy |
| QCM | Custom datasets | QCM accuracy |
| BERTScore | DPO datasets | F1 |
| ROUGE | DPO datasets | ROUGE-L |
| LogProb | DPO datasets | Log-probability ratio |

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- TRL 0.7+
- PEFT 0.7+
- See `requirements.txt` for the full list

## Supported Models

- `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `Qwen/Qwen2-VL-2B-Instruct`

## Logging

- Training runs are logged to W&B (project: `23ComprehensivePipeline`)
- Pipeline logs are saved to the `logs/` directory
- Model weights are saved to `modelweights/`
