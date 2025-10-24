# SmolVLM Sequential Training Pipeline

This document describes the updated benchmark and training pipeline that supports sequential training on multiple datasets with intermediate benchmarking.

## Overview

The pipeline now supports:
1. **Sequential Training**: Train on DPO dataset, then continue training on QCM dataset
2. **Intermediate Benchmarking**: Benchmark after each training phase
3. **Multiple Dataset Support**: DPO and QCM datasets in the same folder structure
4. **Flexible Configuration**: Command-line arguments for all parameters

## Pipeline Architecture

```
Base Model (SmolVLM-500M-Instruct)
    ↓
[Benchmark] → Results: Base Performance
    ↓
Train on DPO Dataset (dpo_dataset.json)
    ↓
[Benchmark] → Results: After DPO Training
    ↓
Train on QCM Dataset (qcm_dataset.json, starting from DPO-trained model)
    ↓
[Benchmark] → Results: After DPO+QCM Training
    ↓
[Compare All Results] → Final Comparison Report
```

## Dataset Structure

The pipeline expects datasets in the `dpo_image_dataset/` folder:

### DPO Dataset Format (`dpo_dataset.json`)
```json
[
  {
    "prompt": "Question or instruction",
    "chosen": "Preferred/correct response",
    "rejected": "Less preferred/incorrect response",
    "image_name": "image_001.png",
    "type": "descriptive|qa"
  }
]
```

### QCM Dataset Format (`qcm_dataset.json`)
```json
[
  {
    "image_name": "image_001.png",
    "type": "qcm",
    "qcm": {
      "question": "Multiple choice question",
      "options": {
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      },
      "correct_answer": "D",
      "explanation": "Why this is correct"
    }
  }
]
```

## Usage

### Sequential Training Pipeline

Run the complete sequential training pipeline with benchmarking:

```bash
# Full pipeline with base model benchmarking
python benchmark_pipeline.py sequential --num-samples 100

# Skip base model benchmarking (faster)
python benchmark_pipeline.py sequential --num-samples 100 --skip-base
```

### Individual Training Scripts

#### Train on DPO Dataset
```bash
# Default training
python finetune_smolvlm_lora.py

# Custom configuration
python finetune_smolvlm_lora.py \
  --base-model HuggingFaceTB/SmolVLM-500M-Instruct \
  --output-dir ./smolvlm-500m-dpo-finetuned \
  --dataset dpo_image_dataset/dpo_dataset.json \
  --num-epochs 3

# Test mode (10 samples only)
python finetune_smolvlm_lora.py --test
```

#### Train on QCM Dataset
```bash
# Default training
python finetune_smolvlm_qcm.py

# Continue from DPO-trained model
python finetune_smolvlm_qcm.py \
  --base-model ./smolvlm-500m-dpo-finetuned \
  --output-dir ./smolvlm-500m-dpo-qcm-finetuned \
  --dataset dpo_image_dataset/qcm_dataset.json \
  --num-epochs 3

# Test mode
python finetune_smolvlm_qcm.py --test
```

### Benchmark Only

Benchmark a single model without training:

```bash
python benchmark_pipeline.py benchmark \
  --model-path ./smolvlm-500m-dpo-finetuned \
  --num-samples 100
```

### Compare Models

Compare two models (base vs fine-tuned):

```bash
python benchmark_pipeline.py compare \
  --base-model HuggingFaceTB/SmolVLM-500M-Instruct \
  --finetuned-model ./smolvlm-500m-dpo-finetuned \
  --num-samples 500
```

## Benchmarks Included

The pipeline runs the following benchmarks:

1. **OCRBench**: OCR capabilities on document images
2. **DocVQA**: Document visual question answering
3. **ChartQA**: Chart and graph understanding
4. **ERP QCM**: Multiple choice questions on ERP interface screenshots
5. **DPO LogProb**: Log probability analysis on DPO preference pairs
6. **BERTScore**: Semantic similarity between generated and reference answers

## Output Structure

Results are saved in `./benchmark_results/`:

- `base_model_<timestamp>.json` - Base model results
- `dpo_model_<timestamp>.json` - After DPO training
- `dpo_qcm_model_<timestamp>.json` - After DPO+QCM training
- `sequential_comparison_<timestamp>.json` - Complete comparison

## Training Configuration

### Memory Optimization
- **4-bit Quantization**: Reduces memory usage significantly
- **LoRA**: Only trains ~1% of parameters
- **Gradient Checkpointing**: Trades compute for memory
- **8-bit Optimizer**: Further memory savings

### Default Hyperparameters
- **Batch Size**: 1 (required for variable image sizes)
- **Gradient Accumulation**: 8 steps (effective batch size = 8)
- **Learning Rate**: 1e-5
- **Epochs**: 3
- **LoRA Rank**: 16
- **LoRA Alpha**: 32

## WandB Integration

Training automatically logs to Weights & Biases:
- Training loss
- Evaluation loss
- Learning rate schedule
- GPU utilization

To disable WandB logging, use `--test` mode or set `WANDB_MODE=disabled`.

## Tips and Best Practices

1. **Start Small**: Use `--test` mode first to verify everything works
2. **Sequential Training**: Always train on DPO first, then QCM
3. **Checkpoint Management**: Keep intermediate checkpoints for comparison
4. **Benchmarking**: Use at least 100 samples for meaningful metrics
5. **Memory**: Requires ~12GB VRAM for training (with optimizations)

## Troubleshooting

### Out of Memory
- Reduce `gradient_accumulation_steps` if needed
- Ensure no other processes are using GPU
- Try reducing image resolution in dataset preprocessing

### Model Loading Errors
- Verify model path is correct
- Ensure LoRA adapters are properly saved
- Check HuggingFace cache if using remote models

### Dataset Errors
- Verify JSON format matches examples above
- Ensure all referenced images exist
- Check image file formats (PNG, JPG supported)

## Future Enhancements

Potential improvements for the pipeline:
- [ ] Add intermediate benchmarking during training (eval steps)
- [ ] Support for mixed precision training
- [ ] Automated hyperparameter tuning
- [ ] Multi-GPU training support
- [ ] More sophisticated data augmentation
