"""
Base Dataset Classes - Abstract base classes for all dataset loaders

Provides:
- BaseDataset: Abstract base for all datasets
- BaseVisionDataset: Abstract base for vision-language datasets
- BaseDPODataset: Abstract base for DPO preference datasets
- Common utilities for image loading, answer masking, and dataset operations
"""

import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

# Default constants
# SmolVLM default: size={"longest_edge": N*512} where N=4, so 2048px
# See: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
DEFAULT_MAX_IMAGE_SIZE = 2048
DEFAULT_PATCH_SIZE = 16
DEFAULT_PLACEHOLDER_SIZE = (512, 512)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing
    
    Attributes:
        max_samples: Maximum number of samples to load (None for all)
        max_image_size: Maximum image dimension - default 2048px (SmolVLM default N=4*512)
        patch_size: Patch size for ensuring divisibility (default 16)
        placeholder_size: Size for placeholder images when loading fails
        cache_images: Whether to cache resized images
        force_patch_divisible: Ensure dimensions divisible by patch_size
        use_fixed_subset: Use deterministic subset selection
        subset_seed: Seed for subset selection
    """
    max_samples: Optional[int] = None
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE  # 2048px (SmolVLM default N=4*512)
    patch_size: int = DEFAULT_PATCH_SIZE
    placeholder_size: Tuple[int, int] = DEFAULT_PLACEHOLDER_SIZE
    cache_images: bool = True
    force_patch_divisible: bool = True
    use_fixed_subset: bool = False
    subset_seed: int = 42


@dataclass
class DatasetItem:
    """Standard dataset item structure"""
    image: Optional[Image.Image] = None
    image_path: Optional[str] = None
    prompt: str = ""
    response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageUtils:
    """Utility class for image processing operations"""
    
    @staticmethod
    def round_to_patch_size(dim: int, patch_size: int = DEFAULT_PATCH_SIZE) -> int:
        """Round dimension to nearest multiple of patch_size"""
        return max(patch_size, (dim // patch_size) * patch_size)
    
    @staticmethod
    def load_image(image_path: Union[str, Path], convert_rgb: bool = True) -> Optional[Image.Image]:
        """Load image from path with error handling"""
        try:
            path = Path(image_path)
            if path.exists():
                image = Image.open(path)
                if convert_rgb:
                    image = image.convert('RGB')
                return image
            else:
                logger.warning(f"Image not found: {image_path}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    @staticmethod
    def create_placeholder(
        size: Tuple[int, int] = DEFAULT_PLACEHOLDER_SIZE,
        color: str = 'white'
    ) -> Image.Image:
        """Create a placeholder image"""
        return Image.new('RGB', size, color=color)
    
    @staticmethod
    def resize_image(
        image: Image.Image,
        max_size: int = DEFAULT_MAX_IMAGE_SIZE,
        force_patch_divisible: bool = True,
        patch_size: int = DEFAULT_PATCH_SIZE
    ) -> Image.Image:
        """
        Resize image with fallback chain for VLM compatibility.
        
        Strategy (matching image_utils.py):
        - Images <= max_size (default 2048px per SmolVLM): Keep size but ensure divisible by patch_size
        - Images > max_size: Resize to max_size preserving aspect ratio
        - Always ensure final dimensions are divisible by patch_size (16) for DPO compatibility
        
        Note: SmolVLM uses size={"longest_edge": N*512} where N=4 by default (2048px).
        You can decrease N to save GPU memory for lower-resolution images or video fine-tuning.
        
        Args:
            image: PIL Image to resize
            max_size: Maximum dimension (default 2048px per SmolVLM)
            force_patch_divisible: Ensure dimensions divisible by patch_size
            patch_size: Patch size for divisibility (default 16)
            
        Returns:
            Resized RGB image
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        orig_width, orig_height = image.size
        longest_edge = max(orig_width, orig_height)
        
        # Determine target dimensions
        if longest_edge <= max_size:
            new_width, new_height = orig_width, orig_height
        else:
            # Resize preserving aspect ratio
            if orig_width > orig_height:
                new_width = max_size
                new_height = int(orig_height * (max_size / orig_width))
            else:
                new_height = max_size
                new_width = int(orig_width * (max_size / orig_height))
        
        # Ensure dimensions are divisible by patch_size
        if force_patch_divisible:
            new_width = ImageUtils.round_to_patch_size(new_width, patch_size)
            new_height = ImageUtils.round_to_patch_size(new_height, patch_size)
        
        # Only resize if dimensions changed
        if new_width != orig_width or new_height != orig_height:
            try:
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            except Exception as e:
                logger.warning(f"Failed to resize image: {e}")
                # Fallback to 512x512
                try:
                    image = image.resize((512, 512), Image.Resampling.LANCZOS)
                except Exception as e2:
                    logger.error(f"All resize attempts failed: {e2}")
        
        return image
    
    @staticmethod
    def get_cache_key(identifier: str, width: int, height: int) -> str:
        """Generate a cache key for images"""
        key_str = f"{identifier}_{width}x{height}"
        return hashlib.md5(key_str.encode()).hexdigest()


class AnswerMaskingMixin:
    """Mixin providing answer masking functionality for SFT training"""
    
    def find_answer_start_position(
        self,
        full_token_list: List[int],
        answer_tokens: List[int],
        tokenizer,
        fallback_ratio: float = 0.9
    ) -> int:
        """
        Find the position where the answer starts in the tokenized sequence.
        
        This is critical for correct SFT training - we only want to train
        on the answer portion, not the prompt.
        
        Args:
            full_token_list: Full tokenized sequence
            answer_tokens: Tokenized answer without special tokens
            tokenizer: Tokenizer for fallback methods
            fallback_ratio: Fallback position as ratio of sequence length
            
        Returns:
            Position where answer starts
        """
        # Method 1: Direct search for answer tokens
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i + len(answer_tokens)] == answer_tokens:
                return i
        
        # Method 2: Search for assistant marker
        try:
            assistant_markers = ["Assistant:", "assistant:", "<|assistant|>"]
            for marker in assistant_markers:
                marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
                for i in range(len(full_token_list) - len(marker_tokens) + 1):
                    if full_token_list[i:i + len(marker_tokens)] == marker_tokens:
                        return i + len(marker_tokens)
        except Exception as e:
            logger.debug(f"Assistant marker search failed: {e}")
        
        # Method 3: Fallback to ratio-based position
        logger.warning(f"Could not find answer position, using fallback ratio {fallback_ratio}")
        return int(len(full_token_list) * fallback_ratio)
    
    def create_masked_labels(
        self,
        input_ids: torch.Tensor,
        answer_start_pos: int,
        min_unmasked_tokens: int = 2
    ) -> torch.Tensor:
        """
        Create labels tensor with prompt tokens masked.
        
        Args:
            input_ids: Full input_ids tensor
            answer_start_pos: Position where answer starts
            min_unmasked_tokens: Minimum tokens to leave unmasked
            
        Returns:
            Labels tensor with masked positions set to -100
        """
        labels = input_ids.clone()
        
        # Handle both 1D and 2D tensors
        if labels.dim() == 1:
            labels[:answer_start_pos] = -100
            unmasked_count = (labels != -100).sum().item()
        else:
            labels[:, :answer_start_pos] = -100
            unmasked_count = (labels[0] != -100).sum().item()
        
        # Validate masking
        if unmasked_count == 0:
            logger.error(f"All tokens masked! answer_start_pos={answer_start_pos}")
            # Emergency fix: unmask last few tokens
            if labels.dim() == 1:
                labels[-min_unmasked_tokens:] = input_ids[-min_unmasked_tokens:]
            else:
                labels[:, -min_unmasked_tokens:] = input_ids[:, -min_unmasked_tokens:]
        elif unmasked_count < min_unmasked_tokens:
            logger.warning(f"Only {unmasked_count} tokens unmasked")
        
        return labels


class BaseDataset(Dataset, ABC):
    """Abstract base class for all datasets"""
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self._data: List[Any] = []
        self._length: int = 0
    
    @abstractmethod
    def load_data(self, source: Union[str, Path, Any]) -> None:
        """Load data from source"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index"""
        pass
    
    def __len__(self) -> int:
        return self._length
    
    @property
    def data(self) -> List[Any]:
        return self._data
    
    def subset(self, indices: List[int]) -> 'BaseDataset':
        """Create a subset of the dataset"""
        # This is a simple implementation; subclasses can override
        subset_data = [self._data[i] for i in indices if i < len(self._data)]
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.config = self.config
        new_dataset._data = subset_data
        new_dataset._length = len(subset_data)
        return new_dataset


class BaseVisionDataset(BaseDataset, AnswerMaskingMixin):
    """Abstract base class for vision-language datasets"""
    
    def __init__(
        self,
        processor,
        config: Optional[DatasetConfig] = None,
        image_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__(config)
        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
    
    def load_image_for_item(
        self,
        item: Dict[str, Any],
        image_key: str = 'image_name'
    ) -> Image.Image:
        """
        Load image for a dataset item
        
        Args:
            item: Dataset item dict
            image_key: Key for image path/name in item
            
        Returns:
            PIL Image (placeholder if loading fails)
        """
        image_name = item.get(image_key, '')
        
        if image_name and self.image_dir:
            image_path = self.image_dir / image_name
            image = ImageUtils.load_image(image_path)
            
            if image is not None:
                return ImageUtils.resize_image(
                    image,
                    max_size=self.config.max_image_size,
                    force_patch_divisible=self.config.force_patch_divisible,
                    patch_size=self.config.patch_size
                )
        
        return ImageUtils.create_placeholder(self.config.placeholder_size)
    
    def process_for_training(
        self,
        image: Image.Image,
        prompt: str,
        response: str
    ) -> Dict[str, torch.Tensor]:
        """
        Process image, prompt, and response for SFT training.
        
        Creates properly masked labels where only the response portion
        is used for loss computation.
        
        Args:
            image: PIL Image
            prompt: User prompt text
            response: Expected response text
            
        Returns:
            Dict with input_ids, attention_mask, pixel_values, labels
        """
        # Create message format
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]
        
        # Process full sequence
        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )
        
        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": self.config.max_image_size}
        )
        
        # Find answer position and create masked labels
        answer_tokens = self.processor.tokenizer.encode(response, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()
        
        answer_start_pos = self.find_answer_start_position(
            full_token_list,
            answer_tokens,
            self.processor.tokenizer
        )
        
        labels = self.create_masked_labels(
            full_inputs["input_ids"],
            answer_start_pos
        )
        
        # Build output dict
        inputs = {}
        for key in full_inputs:
            inputs[key] = full_inputs[key].squeeze(0)
        inputs["labels"] = labels.squeeze(0)
        
        return inputs
    
    @abstractmethod
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format the prompt for an item"""
        pass
    
    @abstractmethod
    def get_response(self, item: Dict[str, Any]) -> str:
        """Get the expected response for an item"""
        pass


class BaseDPODataset(BaseDataset):
    """Abstract base class for DPO (Direct Preference Optimization) datasets"""
    
    def __init__(
        self,
        processor,
        config: Optional[DatasetConfig] = None,
        image_dir: Optional[Union[str, Path]] = None
    ):
        super().__init__(config)
        self.processor = processor
        self.image_dir = Path(image_dir) if image_dir else None
    
    @abstractmethod
    def get_prompt(self, item: Dict[str, Any]) -> str:
        """Get the prompt for an item"""
        pass
    
    @abstractmethod
    def get_chosen(self, item: Dict[str, Any]) -> str:
        """Get the chosen (preferred) response"""
        pass
    
    @abstractmethod
    def get_rejected(self, item: Dict[str, Any]) -> str:
        """Get the rejected response"""
        pass
    
    def load_image_for_item(
        self,
        item: Dict[str, Any],
        image_key: str = 'image_name'
    ) -> Optional[Image.Image]:
        """Load image for a dataset item"""
        image_name = item.get(image_key, '')
        
        if image_name and self.image_dir:
            image_path = self.image_dir / image_name
            image = ImageUtils.load_image(image_path)
            
            if image is not None:
                return ImageUtils.resize_image(
                    image,
                    max_size=self.config.max_image_size,
                    force_patch_divisible=self.config.force_patch_divisible,
                    patch_size=self.config.patch_size
                )
        
        return None
    
    def to_chat_format(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        image: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Convert to TRL's expected chat format for DPO training.
        
        Returns:
            Dict with 'prompt', 'chosen', 'rejected', and optionally 'images'
        """
        result: Dict[str, Any] = {
            'prompt': [{
                "role": "user",
                "content": [
                    {"type": "image", "text": None},
                    {"type": "text", "text": prompt}
                ] if image else [
                    {"type": "text", "text": prompt}
                ]
            }],
            'chosen': [{
                "role": "assistant",
                "content": [{"type": "text", "text": chosen}]
            }],
            'rejected': [{
                "role": "assistant",
                "content": [{"type": "text", "text": rejected}]
            }]
        }
        
        if image:
            result['images'] = [image]
        
        return result


class DatasetRegistry:
    """Registry for dataset classes"""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class"""
        def decorator(dataset_cls):
            cls._registry[name] = dataset_cls
            return dataset_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a registered dataset class by name"""
        return cls._registry.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered dataset names"""
        return list(cls._registry.keys())
    
    @classmethod
    def create(cls, name: str, *args, **kwargs) -> BaseDataset:
        """Create a dataset instance by name"""
        dataset_cls = cls.get(name)
        if dataset_cls is None:
            raise ValueError(f"Unknown dataset: {name}. Available: {cls.list_available()}")
        return dataset_cls(*args, **kwargs)
