"""
Data Collators - Batch collation utilities for training

Provides:
- VisionLanguageDataCollator: Collator for vision-language models
- DPODataCollator: Collator for DPO training format
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def pad_pixel_values(pixel_values_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Pad pixel values to the same shape and stack them.

    Handles variable-sized images by padding to the maximum dimensions in the batch.
    For SmolVLM, pixel_values typically have shape (num_images, channels, height, width)
    or (channels, height, width).

    Args:
        pixel_values_list: List of pixel value tensors

    Returns:
        Stacked and padded tensor
    """
    if not pixel_values_list:
        raise ValueError("Empty pixel_values_list")

    # Get shapes of all tensors
    shapes = [pv.shape for pv in pixel_values_list]

    # Check if all shapes are the same - if so, just stack
    if all(s == shapes[0] for s in shapes):
        return torch.stack(pixel_values_list)

    # Handle different number of dimensions
    ndims = [len(s) for s in shapes]
    if len(set(ndims)) > 1:
        logger.warning(f"Pixel values have different number of dimensions: {ndims}. Attempting to handle.")

    max_ndim = max(ndims)

    # Normalize all tensors to have the same number of dimensions
    normalized = []
    for pv in pixel_values_list:
        while len(pv.shape) < max_ndim:
            pv = pv.unsqueeze(0)
        normalized.append(pv)

    # Get max size for each dimension
    shapes = [pv.shape for pv in normalized]
    max_shape = [max(s[i] for s in shapes) for i in range(max_ndim)]

    # Pad each tensor to max_shape
    padded = []
    for pv in normalized:
        # Calculate padding for each dimension (from last to first)
        pad_sizes = []
        for i in range(max_ndim - 1, -1, -1):
            pad_needed = max_shape[i] - pv.shape[i]
            pad_sizes.extend([0, pad_needed])  # (left, right) for each dim

        if any(p > 0 for p in pad_sizes):
            pv = F.pad(pv, pad_sizes, mode='constant', value=0)
        padded.append(pv)

    return torch.stack(padded)


@dataclass
class VisionLanguageDataCollator:
    """
    Custom data collator for vision-language models.

    Handles:
    - Dynamic padding of text sequences
    - Proper label masking with -100 for padded positions
    - Pixel values padding and stacking (supports variable-sized images)
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dicts from dataset

        Returns:
            Batched tensors dict
        """
        # Extract pixel values first
        pixel_values = [f.pop('pixel_values') for f in features]

        # Find max sequence length
        max_length = max(f['input_ids'].shape[0] for f in features)

        batch = {}

        # Use padding-aware stacking for pixel values (handles different image sizes)
        batch['pixel_values'] = pad_pixel_values(pixel_values)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            seq_len = f['input_ids'].shape[0]
            pad_len = max_length - seq_len

            # Pad input_ids
            input_ids.append(torch.cat([
                f['input_ids'],
                torch.full((pad_len,), self.pad_token_id, dtype=f['input_ids'].dtype)
            ]))

            # Pad attention_mask
            attention_mask.append(torch.cat([
                f['attention_mask'],
                torch.zeros(pad_len, dtype=f['attention_mask'].dtype)
            ]))

            # Pad labels with -100 (ignored in loss)
            labels.append(torch.cat([
                f['labels'],
                torch.full((pad_len,), self.label_pad_token_id, dtype=f['labels'].dtype)
            ]))

        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['labels'] = torch.stack(labels)

        return batch


@dataclass
class VisionLanguageDataCollatorWithPadding:
    """
    Enhanced data collator with configurable padding strategy.
    
    Supports:
    - Left or right padding
    - Maximum length truncation
    - Optional label creation from input_ids
    """
    
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    padding_side: str = "right"
    max_length: Optional[int] = None
    truncation: bool = True
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate with configurable padding"""
        
        # Handle pixel values
        if 'pixel_values' in features[0]:
            pixel_values = [f.pop('pixel_values') for f in features]
        else:
            pixel_values = None
        
        # Get sequence lengths
        lengths = [f['input_ids'].shape[0] for f in features]
        max_seq_len = max(lengths)
        
        # Apply max_length truncation if specified
        if self.max_length and self.truncation:
            max_seq_len = min(max_seq_len, self.max_length)
        
        batch = {}

        if pixel_values:
            batch['pixel_values'] = pad_pixel_values(pixel_values)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            seq_len = min(f['input_ids'].shape[0], max_seq_len) if self.truncation else f['input_ids'].shape[0]
            pad_len = max_seq_len - seq_len
            
            # Truncate if needed
            ids = f['input_ids'][:seq_len]
            mask = f['attention_mask'][:seq_len]
            lab = f['labels'][:seq_len] if 'labels' in f else ids.clone()
            
            if pad_len > 0:
                pad_ids = torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)
                pad_mask = torch.zeros(pad_len, dtype=mask.dtype)
                pad_lab = torch.full((pad_len,), self.label_pad_token_id, dtype=lab.dtype)
                
                if self.padding_side == "right":
                    ids = torch.cat([ids, pad_ids])
                    mask = torch.cat([mask, pad_mask])
                    lab = torch.cat([lab, pad_lab])
                else:
                    ids = torch.cat([pad_ids, ids])
                    mask = torch.cat([pad_mask, mask])
                    lab = torch.cat([pad_lab, lab])
            
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)
        
        batch['input_ids'] = torch.stack(input_ids)
        batch['attention_mask'] = torch.stack(attention_mask)
        batch['labels'] = torch.stack(labels)
        
        return batch


@dataclass
class DPODataCollator:
    """
    Data collator for DPO training format.
    
    Handles the special structure required by TRL's DPOTrainer:
    - prompt, chosen, rejected message lists
    - images list
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate DPO features.
        
        Note: TRL's DPOTrainer typically handles its own collation,
        so this is mainly for custom DPO implementations.
        """
        batch = {
            'prompt': [f['prompt'] for f in features],
            'chosen': [f['chosen'] for f in features],
            'rejected': [f['rejected'] for f in features],
        }
        
        # Handle images if present
        if 'images' in features[0]:
            batch['images'] = [f['images'] for f in features]
        
        return batch


def create_data_collator(
    collator_type: str = "vision_language",
    pad_token_id: int = 0,
    **kwargs
) -> Any:
    """
    Factory function to create data collators.
    
    Args:
        collator_type: Type of collator (vision_language, vision_language_padded, dpo)
        pad_token_id: Token ID for padding
        **kwargs: Additional arguments for specific collators
        
    Returns:
        Data collator instance
    """
    if collator_type == "vision_language":
        return VisionLanguageDataCollator(pad_token_id=pad_token_id)
    elif collator_type == "vision_language_padded":
        return VisionLanguageDataCollatorWithPadding(
            pad_token_id=pad_token_id,
            **kwargs
        )
    elif collator_type == "dpo":
        return DPODataCollator()
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")
