"""
Shared Configuration Setup

This module provides centralized configuration for HuggingFace cache
and model constants used across trainers and evaluators.

Usage:
    from config.setup import setup_hf_cache, BASE_MODEL, HF_CACHE_DIR

    # Call once at module start (before HuggingFace imports)
    setup_hf_cache()
"""

import os
from pathlib import Path

# Base model used throughout the pipeline
BASE_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# HuggingFace cache directory (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent
HF_CACHE_DIR = str(_PROJECT_ROOT / "hf_cache")

_setup_done = False


def setup_hf_cache(cache_dir: str = None) -> str:
    """
    Set up HuggingFace cache environment variables.

    IMPORTANT: Call this before importing any HuggingFace libraries
    (transformers, datasets, peft, etc.)

    Args:
        cache_dir: Optional custom cache directory. If None, uses default.

    Returns:
        The cache directory path that was set.
    """
    global _setup_done, HF_CACHE_DIR

    if _setup_done and cache_dir is None:
        return HF_CACHE_DIR

    if cache_dir:
        HF_CACHE_DIR = os.path.abspath(cache_dir)
    else:
        HF_CACHE_DIR = os.path.abspath(HF_CACHE_DIR)

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")

    _setup_done = True
    return HF_CACHE_DIR


def get_hf_cache_dir() -> str:
    """Get the HuggingFace cache directory, setting up if needed."""
    if not _setup_done:
        setup_hf_cache()
    return HF_CACHE_DIR
