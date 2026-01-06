"""
Dataset Utilities - Helper functions and utilities for dataset operations

Provides:
- Dataset splitting utilities
- Caching utilities
- Memory management
- Dataset statistics
"""

import gc
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset, random_split

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Splitting
# ============================================================================

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into train and eval sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Fraction for training (default 0.9)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    eval_size = dataset_size - train_size
    
    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Split dataset: {train_size} train, {eval_size} eval")
    return train_dataset, eval_dataset


def create_subset(
    dataset: Dataset,
    max_samples: int,
    seed: Optional[int] = None
) -> Dataset:
    """
    Create a subset of a dataset.
    
    Args:
        dataset: Source dataset
        max_samples: Maximum samples to include
        seed: If provided, random subset; otherwise first N samples
        
    Returns:
        Subset dataset
    """
    if max_samples >= len(dataset):
        return dataset
    
    if seed is not None:
        # Random subset
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    else:
        # First N samples
        indices = list(range(max_samples))
    
    return Subset(dataset, indices)


# ============================================================================
# Dataset Caching
# ============================================================================

class DatasetCache:
    """
    Cache manager for processed datasets.
    
    Supports saving and loading datasets to/from disk to avoid
    repeated preprocessing.
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(
        self,
        dataset_path: str,
        dataset_type: str,
        max_samples: Optional[int] = None,
        extra_params: Optional[Dict] = None
    ) -> str:
        """Generate cache key from dataset parameters"""
        key_parts = [dataset_path, dataset_type, str(max_samples)]
        if extra_params:
            key_parts.append(json.dumps(extra_params, sort_keys=True))
        key_str = "_".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cache key"""
        return self.cache_dir / cache_key
    
    def exists(self, cache_key: str) -> bool:
        """Check if cache exists"""
        return self.get_cache_path(cache_key).exists()
    
    def load(self, cache_key: str):
        """Load dataset from cache"""
        from datasets import load_from_disk
        
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading cached dataset: {cache_key}")
            return load_from_disk(str(cache_path))
        return None
    
    def save(self, dataset, cache_key: str) -> None:
        """Save dataset to cache"""
        cache_path = self.get_cache_path(cache_key)
        try:
            logger.info(f"Saving dataset to cache: {cache_key}")
            dataset.save_to_disk(str(cache_path))
        except Exception as e:
            logger.warning(f"Failed to cache dataset: {e}")
    
    def clear(self, cache_key: Optional[str] = None) -> int:
        """
        Clear cache.
        
        Args:
            cache_key: Specific cache to clear, or None for all
            
        Returns:
            Bytes freed
        """
        freed = 0
        
        if cache_key:
            cache_path = self.get_cache_path(cache_key)
            if cache_path.exists():
                freed = self._get_dir_size(cache_path)
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache {cache_key} ({freed / 1e9:.2f} GB)")
        else:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    size = self._get_dir_size(item)
                    shutil.rmtree(item)
                    freed += size
            logger.info(f"Cleared all caches ({freed / 1e9:.2f} GB)")
        
        return freed
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory"""
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())


# ============================================================================
# Memory Management
# ============================================================================

def cleanup_memory() -> None:
    """Clean up memory (CPU and GPU)"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared GPU memory cache")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB"""
    import psutil
    
    process = psutil.Process()
    mem_info = process.memory_info()
    
    result = {
        'process_ram_gb': mem_info.rss / 1e9,
        'process_vms_gb': mem_info.vms / 1e9,
    }
    
    if torch.cuda.is_available():
        result['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        result['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    
    return result


def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage"""
    mem = get_memory_usage()
    msg = f"{prefix} RAM: {mem['process_ram_gb']:.2f} GB"
    if 'gpu_allocated_gb' in mem:
        msg += f", GPU: {mem['gpu_allocated_gb']:.2f} GB"
    logger.info(msg)


# ============================================================================
# Dataset Statistics
# ============================================================================

def compute_dataset_stats(dataset: Dataset, num_samples: int = 100) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to analyze
        
    Returns:
        Dict with statistics
    """
    stats = {
        'total_samples': len(dataset),
        'analyzed_samples': min(num_samples, len(dataset)),
    }
    
    # Sample some items
    indices = list(range(min(num_samples, len(dataset))))
    
    seq_lengths = []
    label_lengths = []
    has_images = 0
    
    for idx in indices:
        try:
            item = dataset[idx]
            
            if 'input_ids' in item:
                seq_lengths.append(len(item['input_ids']))
            
            if 'labels' in item:
                # Count non-masked labels
                labels = item['labels']
                if isinstance(labels, torch.Tensor):
                    label_lengths.append((labels != -100).sum().item())
            
            if 'pixel_values' in item:
                has_images += 1
                
        except Exception as e:
            logger.warning(f"Error analyzing sample {idx}: {e}")
    
    if seq_lengths:
        stats['avg_seq_length'] = sum(seq_lengths) / len(seq_lengths)
        stats['max_seq_length'] = max(seq_lengths)
        stats['min_seq_length'] = min(seq_lengths)
    
    if label_lengths:
        stats['avg_label_length'] = sum(label_lengths) / len(label_lengths)
        stats['max_label_length'] = max(label_lengths)
        stats['min_label_length'] = min(label_lengths)
    
    stats['has_images_ratio'] = has_images / len(indices) if indices else 0
    
    return stats


def validate_dataset(
    dataset: Dataset,
    num_samples: int = 10,
    required_keys: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate dataset structure and contents.
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of samples to check
        required_keys: Keys that must be present in each item
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    required_keys = required_keys or ['input_ids', 'attention_mask', 'labels']
    errors = []
    
    if len(dataset) == 0:
        errors.append("Dataset is empty")
        return False, errors
    
    for idx in range(min(num_samples, len(dataset))):
        try:
            item = dataset[idx]
            
            # Check required keys
            for key in required_keys:
                if key not in item:
                    errors.append(f"Sample {idx}: Missing key '{key}'")
            
            # Check for all-masked labels
            if 'labels' in item:
                labels = item['labels']
                if isinstance(labels, torch.Tensor):
                    unmasked = (labels != -100).sum().item()
                    if unmasked == 0:
                        errors.append(f"Sample {idx}: All labels masked")
                    elif unmasked < 2:
                        errors.append(f"Sample {idx}: Only {unmasked} labels unmasked")
            
            # Check input_ids shape
            if 'input_ids' in item:
                ids = item['input_ids']
                if isinstance(ids, torch.Tensor) and ids.numel() == 0:
                    errors.append(f"Sample {idx}: Empty input_ids")
                    
        except Exception as e:
            errors.append(f"Sample {idx}: Error accessing - {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


# ============================================================================
# Dataset Info
# ============================================================================

def print_dataset_info(dataset: Dataset, name: str = "Dataset") -> None:
    """Print formatted dataset information"""
    print(f"\n{'='*60}")
    print(f"{name} Information")
    print(f"{'='*60}")
    print(f"Total samples: {len(dataset)}")
    
    # Get stats
    stats = compute_dataset_stats(dataset)
    
    if 'avg_seq_length' in stats:
        print(f"Sequence length: {stats['min_seq_length']}-{stats['max_seq_length']} "
              f"(avg: {stats['avg_seq_length']:.1f})")
    
    if 'avg_label_length' in stats:
        print(f"Label tokens: {stats['min_label_length']}-{stats['max_label_length']} "
              f"(avg: {stats['avg_label_length']:.1f})")
    
    print(f"Has images: {stats['has_images_ratio']*100:.0f}%")
    
    # Validate
    is_valid, errors = validate_dataset(dataset)
    if is_valid:
        print("✓ Validation passed")
    else:
        print(f"✗ Validation failed with {len(errors)} errors:")
        for err in errors[:5]:  # Show first 5 errors
            print(f"  - {err}")
    
    print(f"{'='*60}\n")
