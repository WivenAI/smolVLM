"""
Model Loading Utilities - Unified model loading for all training strategies

Provides:
- get_bnb_4bit_config: BitsAndBytes 4-bit quantization config
- get_lora_config: LoRA configuration for QLoRA training
- load_model_qlora: Load model with QLoRA for SFT
- load_model_qlora_dpo: Load model with QLoRA for DPO (do_image_splitting=False)
- load_model_full_ft: Load model for full fine-tuning (no quantization)
- resolve_cache_dir: Resolve cache directory path
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config.setup import BASE_MODEL

logger = logging.getLogger(__name__)


def resolve_cache_dir(cache_dir: Optional[str], config: dict = None) -> Optional[str]:
    """
    Resolve cache directory path, handling relative paths.

    Args:
        cache_dir: Cache directory path (relative or absolute)
        config: Optional config dict to extract cache_dir from

    Returns:
        Resolved absolute path or None
    """
    if config and not cache_dir:
        cache_dir = config.get("model", {}).get("cache_dir", None)

    if cache_dir and not os.path.isabs(cache_dir):
        # Resolve relative to project root
        cache_dir = str(Path(__file__).parent.parent / cache_dir)

    if cache_dir:
        logger.info(f"Using cache_dir: {cache_dir}")

    return cache_dir


def get_bnb_4bit_config() -> BitsAndBytesConfig:
    """
    Get BitsAndBytes 4-bit quantization config for QLoRA.

    Uses NF4 quantization with double quantization and bfloat16 compute dtype
    for optimal memory efficiency while maintaining training quality.

    Returns:
        BitsAndBytesConfig configured for 4-bit QLoRA training
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> LoraConfig:
    """
    Get LoRA configuration for QLoRA training.

    Default targets attention, MLP, and lm_head layers for maximum
    adaptation capacity with vision-language models.

    Args:
        r: LoRA rank (default 32 for good capacity)
        lora_alpha: LoRA alpha scaling (default 64)
        lora_dropout: Dropout rate (default 0.05)
        target_modules: List of modules to apply LoRA to (default: attention+MLP+lm_head)

    Returns:
        LoraConfig for QLoRA training
    """
    if target_modules is None:
        target_modules = [
            # Attention layers (Q, K, V, O projections)
            "q_proj", "v_proj", "k_proj", "o_proj",
            # MLP layers (most knowledge stored here)
            "gate_proj", "up_proj", "down_proj",
            # Output projection for language modeling
            "lm_head"
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )


def load_model_qlora(
    base_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    print_trainable_params: bool = True
) -> Tuple[Any, Any]:
    """
    Load model with QLoRA for SFT training.

    Loads model with 4-bit quantization and applies LoRA adapters.

    Args:
        base_model: Model name/path (default BASE_MODEL from config)
        cache_dir: HuggingFace cache directory
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        print_trainable_params: Whether to print trainable parameter count

    Returns:
        Tuple of (model, processor)
    """
    if base_model is None:
        base_model = BASE_MODEL

    logger.info(f"Loading model with QLoRA: {base_model}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    # Load model with 4-bit quantization
    bnb_config = get_bnb_4bit_config()
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = get_lora_config(r=lora_r, lora_alpha=lora_alpha)
    model = get_peft_model(model, lora_config)

    if print_trainable_params:
        model.print_trainable_parameters()

    return model, processor


def load_model_qlora_dpo(
    base_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    print_trainable_params: bool = True
) -> Tuple[Any, Any]:
    """
    Load model with QLoRA for DPO training.

    Same as load_model_qlora but with do_image_splitting=False in processor,
    which is required for VLM DPO training per TRL documentation.

    Args:
        base_model: Model name/path (default BASE_MODEL from config)
        cache_dir: HuggingFace cache directory
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling
        print_trainable_params: Whether to print trainable parameter count

    Returns:
        Tuple of (model, processor)
    """
    if base_model is None:
        base_model = BASE_MODEL

    logger.info(f"Loading model with QLoRA for DPO: {base_model}")

    # Load processor with do_image_splitting=False (required for VLM DPO)
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=cache_dir,
        do_image_splitting=False  # Required for VLM DPO training
    )

    # Load model with 4-bit quantization
    bnb_config = get_bnb_4bit_config()
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = get_lora_config(r=lora_r, lora_alpha=lora_alpha)
    model = get_peft_model(model, lora_config)

    if print_trainable_params:
        model.print_trainable_parameters()

    return model, processor


def load_model_full_ft(
    base_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_gradient_checkpointing: bool = True
) -> Tuple[Any, Any]:
    """
    Load model for full fine-tuning (no quantization, no LoRA).

    All model parameters will be trainable. Uses bfloat16 for memory
    efficiency while maintaining training precision.

    Args:
        base_model: Model name/path (default BASE_MODEL from config)
        cache_dir: HuggingFace cache directory
        enable_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Tuple of (model, processor)
    """
    if base_model is None:
        base_model = BASE_MODEL

    logger.info(f"[FULL_FT] Loading model for FULL fine-tuning: {base_model}")

    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    # Load model WITHOUT quantization for full fine-tuning
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    # Enable gradient checkpointing for memory efficiency
    if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"[FULL_FT] Full fine-tuning - ALL parameters trainable:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, processor


def load_model_full_ft_dpo(
    base_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    enable_gradient_checkpointing: bool = True
) -> Tuple[Any, Any]:
    """
    Load model for full fine-tuning DPO (no quantization, no LoRA).

    Same as load_model_full_ft but with do_image_splitting=False in processor,
    which is required for VLM DPO training per TRL documentation.

    Args:
        base_model: Model name/path (default BASE_MODEL from config)
        cache_dir: HuggingFace cache directory
        enable_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Tuple of (model, processor)
    """
    if base_model is None:
        base_model = BASE_MODEL

    logger.info(f"[FULL_FT-DPO] Loading model for FULL fine-tuning DPO: {base_model}")

    # Load processor with do_image_splitting=False (required for VLM DPO)
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=cache_dir,
        do_image_splitting=False  # Required for VLM DPO training
    )

    # Load model WITHOUT quantization for full fine-tuning
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )

    # Enable gradient checkpointing for memory efficiency
    if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"[FULL_FT-DPO] Full fine-tuning DPO - ALL parameters trainable:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, processor
