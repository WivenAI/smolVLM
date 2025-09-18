# SmolVLM Fine-tuning

This repository contains code for fine-tuning the HuggingFaceTB/SmolVLM-256M-Instruct model on a small Visual Question Answering (VQA) dataset.

## Files

- `finetune_smolvlm.py` - Main fine-tuning script
- `test_finetuned_model.py` - Script to test the fine-tuned model
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the fine-tuning script:
```bash
python finetune_smolvlm.py
```

3. Test the fine-tuned model:
```bash
python test_finetuned_model.py
```

## Dataset

The script uses the VQA-RAD dataset (first 50 samples) for demonstration. VQA-RAD is a small medical visual question answering dataset with 2,248 question-answer pairs and 315 radiology images.

## Model Details

- **Base Model**: HuggingFaceTB/SmolVLM-256M-Instruct
- **Parameters**: 256M parameters
- **Training**: 2 epochs with gradient accumulation
- **Memory**: Optimized for systems with limited GPU memory

## Training Configuration

- Batch size: 1 per device
- Gradient accumulation: 4 steps
- Learning rate: 5e-5
- FP16 precision (if CUDA available)
- Gradient checkpointing enabled

## Output

The fine-tuned model will be saved in the `./smolvlm-finetuned/` directory.