"""
QCM Dataset Classes - Multiple Choice Question datasets for ERP training

Provides:
- QCMDataset: Standard QCM dataset for SFT training
- QCMDPODataset: QCM dataset formatted for DPO training
- QCMCombinedDataset: Combined multiple QCM datasets
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

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


@DatasetRegistry.register("qcm")
class QCMDataset(BaseVisionDataset):
    """
    Dataset for QCM (Multiple Choice Questions) training.
    
    Supports two JSON formats:
    1. Flat format: [{"question": ..., "options": {...}, "correct_answer": ..., "image_name": ...}]
    2. Nested format: [{"qcm": {"question": ..., "options": {...}, "correct_answer": ...}, "image_name": ...}]
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
        """Load QCM data from JSON file"""
        with open(source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self._original_items = raw_data
        
        # Handle nested 'qcm' structure
        if raw_data and isinstance(raw_data[0], dict) and 'qcm' in raw_data[0]:
            self._data = [item['qcm'] for item in raw_data]
        else:
            self._data = raw_data
        
        # Apply max_samples limit
        if self.config.max_samples and self.config.max_samples < len(self._data):
            self._data = self._data[:self.config.max_samples]
            self._original_items = self._original_items[:self.config.max_samples]
        
        self._length = len(self._data)
        logger.info(f"Loaded {self._length} QCM examples from {source}")
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format QCM prompt with question and options"""
        # Handle both nested and flat structures
        qcm_data = item.get('qcm', item)
        question = qcm_data['question']
        options = qcm_data['options']
        
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        return f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter of the correct option:"
    
    def get_response(self, item: Dict[str, Any]) -> str:
        """Get the correct answer letter"""
        qcm_data = item.get('qcm', item)
        return qcm_data['correct_answer']
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._data[idx]
        original_item = self._original_items[idx]
        
        # Load image
        image = self.load_image_for_item(original_item)
        
        # Format prompt and get response
        prompt = self.format_prompt(item)
        response = self.get_response(item)
        
        # Process for training
        return self.process_for_training(image, prompt, response)
    
    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw item for evaluation purposes"""
        item = self._data[idx]
        original_item = self._original_items[idx]
        
        qcm_data = item.get('qcm', item)
        return {
            'question': qcm_data['question'],
            'options': qcm_data['options'],
            'correct_answer': qcm_data['correct_answer'],
            'image_name': original_item.get('image_name', '')
        }


@DatasetRegistry.register("qcm_dpo")
class QCMDPODataset(BaseDPODataset):
    """
    QCM Dataset formatted for DPO training.
    
    Creates preference pairs where:
    - Chosen = correct answer letter
    - Rejected = random incorrect answer letter
    """
    
    def __init__(
        self,
        json_path: Union[str, Path],
        image_dir: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None,
        seed: int = 42
    ):
        super().__init__(processor, config, image_dir)
        self.json_path = Path(json_path)
        self.rng = random.Random(seed)
        self.load_data(json_path)
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load and convert QCM data to DPO format"""
        with open(source, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        dpo_data = []
        skipped = 0
        
        for item in raw_data:
            image_name = item.get('image_name', '')
            
            # Skip items without images
            if not image_name:
                skipped += 1
                continue
            
            # Check image exists
            if self.image_dir:
                image_path = self.image_dir / image_name
                if not image_path.exists():
                    skipped += 1
                    continue
            
            # Extract QCM data
            qcm_data = item.get('qcm', item)
            question = qcm_data.get('question', '')
            options = qcm_data.get('options', {})
            correct_answer = qcm_data.get('correct_answer', '')
            
            if not question or not options or not correct_answer:
                skipped += 1
                continue
            
            # Create DPO item
            dpo_data.append({
                'image_name': image_name,
                'question': question,
                'options': options,
                'correct_answer': correct_answer
            })
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(dpo_data):
            if self.config.use_fixed_subset:
                indices = list(range(len(dpo_data)))
                self.rng.shuffle(indices)
                dpo_data = [dpo_data[i] for i in sorted(indices[:self.config.max_samples])]
            else:
                dpo_data = dpo_data[:self.config.max_samples]
        
        self._data = dpo_data
        self._length = len(dpo_data)
        
        if skipped > 0:
            logger.warning(f"Skipped {skipped} QCM items")
        logger.info(f"Loaded {self._length} QCM items for DPO")
    
    def get_prompt(self, item: Dict[str, Any]) -> str:
        """Format QCM prompt"""
        question = item['question']
        options = item['options']
        options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
        return f"{question}\n\nOptions:\n{options_text}\n\nAnswer with the letter of the correct option:"
    
    def get_chosen(self, item: Dict[str, Any]) -> str:
        """Get correct answer (chosen response)"""
        return item['correct_answer']
    
    def get_rejected(self, item: Dict[str, Any]) -> str:
        """Get random incorrect answer (rejected response)"""
        correct = item['correct_answer']
        options = item['options']
        wrong_options = [key for key in options.keys() if key != correct]
        return self.rng.choice(wrong_options) if wrong_options else "X"
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._data[idx]
        
        # Load image
        image = self.load_image_for_item(item)
        
        # Get prompt and responses
        prompt = self.get_prompt(item)
        chosen = self.get_chosen(item)
        rejected = self.get_rejected(item)
        
        return self.to_chat_format(prompt, chosen, rejected, image)


class QCMCombinedDataset(torch.utils.data.ConcatDataset):
    """
    Combined dataset from multiple QCM JSON files.
    
    Useful for training on multiple QCM sources (e.g., Gemini + Nova).
    """
    
    def __init__(
        self,
        json_paths: List[Union[str, Path]],
        image_dir: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None
    ):
        datasets = []
        
        for json_path in json_paths:
            ds = QCMDataset(json_path, image_dir, processor, config)
            datasets.append(ds)
            logger.info(f"Added {len(ds)} samples from {Path(json_path).name}")
        
        super().__init__(datasets)
        logger.info(f"Combined dataset: {len(self)} total samples")


# Factory function for backward compatibility
def create_qcm_dataset(
    json_path: str,
    image_dir: str,
    processor,
    max_samples: int = None,
    for_dpo: bool = False
) -> Union[QCMDataset, QCMDPODataset]:
    """
    Factory function to create QCM dataset.
    
    Args:
        json_path: Path to QCM JSON file
        image_dir: Directory containing images
        processor: VLM processor
        max_samples: Maximum number of samples
        for_dpo: If True, return DPO-formatted dataset
        
    Returns:
        QCMDataset or QCMDPODataset instance
    """
    config = DatasetConfig(max_samples=max_samples)
    
    if for_dpo:
        return QCMDPODataset(json_path, image_dir, processor, config)
    return QCMDataset(json_path, image_dir, processor, config)
