"""
DPO Dataset Classes - Direct Preference Optimization datasets

Provides:
- DPODataset: Standard DPO dataset with chosen/rejected pairs
- DPOSFTDataset: DPO dataset formatted for SFT (using only chosen responses)
- LazyDPODataset: DPO dataset with lazy image loading for memory efficiency
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Iterator, Tuple

import torch
from PIL import Image

from .base_dataset import (
    BaseVisionDataset,
    BaseDPODataset,
    DatasetConfig,
    DatasetRegistry,
    ImageUtils
)

logger = logging.getLogger(__name__)


@DatasetRegistry.register("dpo")
class DPODataset(BaseDPODataset):
    """
    Standard DPO Dataset for preference learning.
    
    Expected JSON format:
    [
        {
            "image_name": "image.png",
            "prompt": "What is shown in this image?",
            "chosen": "A cat sitting on a mat.",
            "rejected": "A dog running in a park."
        },
        ...
    ]
    """
    
    def __init__(
        self,
        json_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None
    ):
        super().__init__(processor, config, image_dir)
        self.json_path = Path(json_path)
        self.load_data(json_path)
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load DPO data from JSON file"""
        with open(source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        valid_data = []
        skipped_missing = 0
        skipped_incomplete = 0
        
        for item in raw_data:
            # Validate required fields
            if not all(key in item for key in ['prompt', 'chosen', 'rejected']):
                skipped_incomplete += 1
                continue
            
            # Check image if specified
            image_name = item.get('image_name', '')
            if image_name and self.image_dir:
                image_path = self.image_dir / image_name
                if not image_path.exists():
                    skipped_missing += 1
                    continue
            
            valid_data.append(item)
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(valid_data):
            if self.config.use_fixed_subset:
                rng = random.Random(self.config.subset_seed)
                indices = list(range(len(valid_data)))
                rng.shuffle(indices)
                valid_data = [valid_data[i] for i in sorted(indices[:self.config.max_samples])]
            else:
                valid_data = valid_data[:self.config.max_samples]
        
        self._data = valid_data
        self._length = len(valid_data)
        
        if skipped_missing + skipped_incomplete > 0:
            logger.warning(f"Skipped {skipped_missing + skipped_incomplete} items "
                         f"({skipped_missing} missing images, {skipped_incomplete} incomplete)")
        logger.info(f"Loaded {self._length} DPO examples")
    
    def get_prompt(self, item: Dict[str, Any]) -> str:
        return item['prompt']
    
    def get_chosen(self, item: Dict[str, Any]) -> str:
        return item['chosen']
    
    def get_rejected(self, item: Dict[str, Any]) -> str:
        return item['rejected']
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._data[idx]
        
        # Load image
        image = self.load_image_for_item(item)
        
        # Get prompt and responses
        prompt = self.get_prompt(item)
        chosen = self.get_chosen(item)
        rejected = self.get_rejected(item)
        
        return self.to_chat_format(prompt, chosen, rejected, image)
    
    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw item for evaluation purposes"""
        return self._data[idx].copy()


@DatasetRegistry.register("dpo_sft")
class DPOSFTDataset(BaseVisionDataset):
    """
    DPO Dataset formatted for SFT training.
    
    Uses only the chosen responses as training targets,
    ignoring the rejected responses. Useful for:
    - Pre-training before DPO
    - Training on preference data without contrastive loss
    """
    
    def __init__(
        self,
        json_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None
    ):
        super().__init__(processor, config, image_dir)
        self.json_path = Path(json_path)
        self.load_data(json_path)
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load DPO data from JSON file"""
        with open(source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        valid_data = []
        skipped = 0
        
        for item in raw_data:
            # Validate required fields
            if not all(key in item for key in ['prompt', 'chosen']):
                skipped += 1
                continue
            
            # Check image
            image_name = item.get('image_name', '')
            if image_name and self.image_dir:
                image_path = self.image_dir / image_name
                if not image_path.exists():
                    skipped += 1
                    continue
            
            valid_data.append(item)
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(valid_data):
            valid_data = valid_data[:self.config.max_samples]
        
        self._data = valid_data
        self._length = len(valid_data)
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} DPO items")
        logger.info(f"Loaded {self._length} DPO examples for SFT")
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        return item['prompt']
    
    def get_response(self, item: Dict[str, Any]) -> str:
        return item['chosen']
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._data[idx]
        
        # Load image
        image = self.load_image_for_item(item)
        
        # Process for training
        prompt = self.format_prompt(item)
        response = self.get_response(item)
        
        return self.process_for_training(image, prompt, response)


class LazyDPODataset(BaseDPODataset):
    """
    DPO Dataset with lazy image loading.
    
    Images are loaded on-demand rather than all at once,
    significantly reducing memory usage for large datasets.
    
    This class stores image paths and loads images only when
    __getitem__ is called, making it suitable for:
    - Large datasets that don't fit in memory
    - Distributed training where each worker loads its own data
    - Streaming/iterative training
    """
    
    def __init__(
        self,
        json_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None
    ):
        super().__init__(processor, config, image_dir)
        self.json_path = Path(json_path)
        self._image_paths: List[Optional[str]] = []
        self.load_data(json_path)
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load DPO data, storing only paths (not images)"""
        with open(source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        valid_data = []
        image_paths = []
        skipped = 0
        
        for item in raw_data:
            # Validate required fields
            if not all(key in item for key in ['prompt', 'chosen', 'rejected']):
                skipped += 1
                continue
            
            # Store image path (don't load yet)
            image_name = item.get('image_name', '')
            image_path_str = None
            
            if image_name and self.image_dir:
                image_path = self.image_dir / image_name
                if image_path.exists():
                    image_path_str = str(image_path)
                else:
                    skipped += 1
                    continue
            
            valid_data.append({
                'prompt': item['prompt'],
                'chosen': item['chosen'],
                'rejected': item['rejected'],
            })
            image_paths.append(image_path_str)
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(valid_data):
            valid_data = valid_data[:self.config.max_samples]
            image_paths = image_paths[:self.config.max_samples]
        
        self._data = valid_data
        self._image_paths = image_paths
        self._length = len(valid_data)
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} DPO items")
        logger.info(f"Loaded {self._length} DPO examples (lazy loading)")
    
    def get_prompt(self, item: Dict[str, Any]) -> str:
        return item['prompt']
    
    def get_chosen(self, item: Dict[str, Any]) -> str:
        return item['chosen']
    
    def get_rejected(self, item: Dict[str, Any]) -> str:
        return item['rejected']
    
    def _load_image_at_index(self, idx: int) -> Image.Image:
        """Load image for index on-demand. Raises exception if image cannot be loaded."""
        image_path = self._image_paths[idx]

        if image_path is None:
            raise ValueError(f"DPO dataset item at index {idx} has no image path - "
                           f"all DPO samples must have valid images")

        image = ImageUtils.load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load DPO image at index {idx}: {image_path}")

        image = ImageUtils.resize_image(
            image,
            max_size=self.config.max_image_size,
            force_patch_divisible=self.config.force_patch_divisible,
            patch_size=self.config.patch_size
        )
        if image is None:
            raise ValueError(f"Failed to resize DPO image at index {idx}: {image_path}")

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._data[idx]

        # Load image on-demand - raises exception if image cannot be loaded
        image = self._load_image_at_index(idx)

        # Get prompt and responses
        prompt = self.get_prompt(item)
        chosen = self.get_chosen(item)
        rejected = self.get_rejected(item)
        
        return self.to_chat_format(prompt, chosen, rejected, image)


class DPODatasetIterator:
    """
    Iterator for DPO datasets that handles image loading and error tracking.
    
    Useful for evaluation and inference where you need to iterate
    through all items while tracking skip counts.
    """
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        image_dir: Union[str, Path],
        evaluator_name: str = "Evaluator",
        max_image_size: int = 2048  # SmolVLM default (N=4*512)
    ):
        self.dataset = dataset
        self.image_dir = Path(image_dir) if isinstance(image_dir, str) else image_dir
        self.evaluator_name = evaluator_name
        self.max_image_size = max_image_size
        self.skipped_missing_image = 0
        self.skipped_error = 0
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """Iterate over dataset, yielding (item, image) tuples"""
        for item in self.dataset:
            try:
                image_name = item.get('image_name', '')
                if not image_name:
                    self.skipped_missing_image += 1
                    continue
                
                image_path = self.image_dir / image_name
                image = ImageUtils.load_image(image_path)
                
                if image is None:
                    self.skipped_missing_image += 1
                    continue
                
                image = ImageUtils.resize_image(image, self.max_image_size)
                yield item, image
                
            except Exception as e:
                logger.warning(f"Error processing example: {e}")
                self.skipped_error += 1
                continue
    
    def get_skip_counts(self) -> Tuple[int, int]:
        """Get counts of skipped samples"""
        return self.skipped_missing_image, self.skipped_error
    
    def log_skip_summary(self) -> None:
        """Log summary of skipped samples"""
        total = self.skipped_missing_image + self.skipped_error
        if total > 0:
            logger.warning(
                f"{self.evaluator_name}: Skipped {total} samples "
                f"({self.skipped_missing_image} missing images, {self.skipped_error} errors)"
            )


# Utility functions
def load_dpo_dataset(
    dataset_path: str,
    max_samples: Optional[int] = None,
    use_fixed_subset: bool = False,
    subset_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load DPO dataset from JSON file (utility function for backward compatibility).
    
    Args:
        dataset_path: Path to DPO JSON file
        max_samples: Maximum samples to return
        use_fixed_subset: Use fixed random subset for reproducibility
        subset_seed: Seed for subset selection
        
    Returns:
        List of dataset items
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples and max_samples < len(dataset):
        if use_fixed_subset:
            rng = random.Random(subset_seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)
            dataset = [dataset[i] for i in sorted(indices[:max_samples])]
            logger.info(f"Using fixed subset of {len(dataset)} samples (seed={subset_seed})")
        else:
            dataset = dataset[:max_samples]
    
    return dataset


# Factory function
def create_dpo_dataset(
    json_path: str,
    image_dir: str,
    processor,
    max_samples: int = None,
    for_sft: bool = False,
    lazy_loading: bool = False
) -> Union[DPODataset, DPOSFTDataset, LazyDPODataset]:
    """
    Factory function to create DPO dataset.
    
    Args:
        json_path: Path to DPO JSON file
        image_dir: Directory containing images
        processor: VLM processor
        max_samples: Maximum samples
        for_sft: If True, return SFT-formatted dataset
        lazy_loading: If True, use lazy image loading
        
    Returns:
        Appropriate DPO dataset instance
    """
    config = DatasetConfig(max_samples=max_samples)
    
    if for_sft:
        return DPOSFTDataset(json_path, image_dir, processor, config)
    elif lazy_loading:
        return LazyDPODataset(json_path, image_dir, processor, config)
    return DPODataset(json_path, image_dir, processor, config)
