"""
Benchmark Dataset Classes - Standard VQA benchmark datasets

Provides:
- BenchmarkDataset: Generic benchmark dataset loader
- DocVQADataset: Document Visual Question Answering
- ChartQADataset: Chart Question Answering
- OCRBenchDataset: OCR Benchmark
- BenchmarkDPODataset: Benchmark formatted for DPO training
"""

import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Iterator

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


# Benchmark configurations
# Supports both HuggingFace datasets and local JSON files
BENCHMARK_CONFIGS = {
    "docvqa": {
        "hf_name": "nielsr/docvqa_1200_examples",
        "split": "train",
        "image_key": "image",
        "question_key": "query",  # For local JSON it's nested dict with 'en' key
        "answer_key": "answers",
        # Local JSON field mappings
        "local_question_key": "query",
        "local_answer_key": "answers",
        "local_image_key": "image_path",
    },
    "ocrbench": {
        "hf_name": "echo840/OCRBench",
        "split": "test",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answer",
        # Local JSON field mappings
        "local_question_key": "question",
        "local_answer_key": "answer",
        "local_image_key": "image_path",
    },
    "chartqa": {
        "hf_name": "HuggingFaceM4/ChartQA",
        "split": "test",
        "image_key": "image",
        "question_key": "query",
        "answer_key": "label",
        # Local JSON field mappings
        "local_question_key": "query",
        "local_answer_key": "label",
        "local_image_key": "image_path",
    }
}


class BenchmarkMixin:
    """Mixin providing common benchmark functionality"""
    
    @staticmethod
    def extract_question(item: Dict[str, Any], question_key: str = "question") -> str:
        """Extract question from various formats"""
        # Try the specified key first
        if question_key in item:
            value = item[question_key]
            if isinstance(value, dict):
                # Handle multi-language queries (e.g., DocVQA with 'en', 'de', 'fr' keys)
                return value.get('en', str(next(iter(value.values()), '')))
            return str(value)
        
        # Fallback keys for different dataset formats
        for key in ['query', 'question', 'text', 'prompt']:
            if key in item:
                value = item[key]
                if isinstance(value, dict):
                    return value.get('en', str(next(iter(value.values()), '')))
                return str(value)
        
        return "What do you see in this image?"
    
    @staticmethod
    def extract_answer(item: Dict[str, Any], answer_key: str = "answers") -> str:
        """Extract answer from various formats"""
        for key in [answer_key, 'answers', 'answer', 'label']:
            if key in item:
                value = item[key]
                if isinstance(value, list) and len(value) > 0:
                    return str(value[0])
                elif value is not None:
                    return str(value)
        return "Unknown"
    
    @staticmethod
    def extract_all_answers(item: Dict[str, Any], answer_key: str = "answers") -> List[str]:
        """Extract all valid answers (for evaluation with multiple ground truths)"""
        for key in [answer_key, 'answers', 'answer', 'label']:
            if key in item:
                value = item[key]
                if isinstance(value, list):
                    return [str(v) for v in value if v is not None]
                elif value is not None:
                    return [str(value)]
        return ["Unknown"]
    
    @staticmethod
    def extract_image(
        item: Dict[str, Any],
        image_key: str = "image",
        max_size: int = 2048
    ) -> Optional[Image.Image]:
        """Extract and process image from dataset item"""
        # First check for image_path (local JSON format)
        image_path = item.get('image_path')
        if image_path and Path(image_path).exists():
            image = ImageUtils.load_image(Path(image_path))
            if image is not None:
                return ImageUtils.resize_image(image, max_size)
        
        # Then try direct image keys (HuggingFace format)
        for key in [image_key, 'image', 'img']:
            if key in item and item[key] is not None:
                image = item[key]
                
                # Handle PIL Image
                if isinstance(image, Image.Image):
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    return ImageUtils.resize_image(image, max_size)
        
        return None


@DatasetRegistry.register("benchmark")
class BenchmarkDataset(BaseVisionDataset, BenchmarkMixin):
    """
    Generic Benchmark Dataset for training on VQA benchmarks.
    
    Supports:
    - HuggingFace datasets (DocVQA, ChartQA, OCRBench)
    - Local JSON files with cached images
    
    JSON format expected:
    [
        {
            "question": "...",  # or "query" for ChartQA/DocVQA
            "answer": ["..."],  # or "label" for ChartQA
            "image_path": "/path/to/image.jpg"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        benchmark_name_or_path: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None,
        benchmark_type: Optional[str] = None  # "docvqa", "chartqa", "ocrbench"
    ):
        super().__init__(processor, config)
        self.benchmark_name = benchmark_type or self._infer_benchmark_type(benchmark_name_or_path)
        self.hf_dataset = None
        self._local_data: Optional[List[Dict[str, Any]]] = None
        self._is_local = False
        self.load_data(benchmark_name_or_path)
    
    def _infer_benchmark_type(self, source: Union[str, Path]) -> str:
        """Infer benchmark type from filename or source"""
        source_str = str(source).lower()
        
        if "docvqa" in source_str:
            return "docvqa"
        elif "chartqa" in source_str:
            return "chartqa"
        elif "ocrbench" in source_str:
            return "ocrbench"
        
        return "generic"
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load benchmark dataset from HuggingFace or local JSON"""
        import json
        
        source_path = Path(source)
        
        # Check if it's a local JSON file
        if source_path.exists() and source_path.suffix == '.json':
            self._load_from_json(source_path)
        else:
            self._load_from_huggingface(str(source))
    
    def _load_from_json(self, json_path: Path) -> None:
        """Load from local JSON file"""
        import json
        
        logger.info(f"Loading benchmark from local JSON: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self._local_data = json.load(f)
        
        self._is_local = True
        
        # Get config for this benchmark type
        benchmark_config = BENCHMARK_CONFIGS.get(self.benchmark_name, {})
        self._image_key = benchmark_config.get('local_image_key', 'image_path')
        self._question_key = benchmark_config.get('local_question_key', 'question')
        self._answer_key = benchmark_config.get('local_answer_key', 'answer')
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(self._local_data):
            self._local_data = self._local_data[:self.config.max_samples]
        
        self._length = len(self._local_data)
        logger.info(f"Loaded {self._length} samples from local JSON ({self.benchmark_name})")
    
    def _load_from_huggingface(self, source: str) -> None:
        """Load from HuggingFace datasets"""
        from datasets import load_dataset
        
        benchmark_config = BENCHMARK_CONFIGS.get(source, {})
        
        hf_name = benchmark_config.get('hf_name', source)
        split = benchmark_config.get('split', 'test')
        
        logger.info(f"Loading {source} dataset from HuggingFace...")
        
        self.hf_dataset = load_dataset(
            hf_name,
            split=split,
            trust_remote_code=True
        )
        
        self._is_local = False
        
        # Store config for extraction
        self._image_key = benchmark_config.get('image_key', 'image')
        self._question_key = benchmark_config.get('question_key', 'question')
        self._answer_key = benchmark_config.get('answer_key', 'answers')
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(self.hf_dataset):
            self.hf_dataset = self.hf_dataset.select(range(self.config.max_samples))
        
        self._length = len(self.hf_dataset)
        logger.info(f"Loaded {self._length} samples from HuggingFace ({source})")
    
    def _get_item_at(self, idx: int) -> Dict[str, Any]:
        """Get raw item at index from either local or HF dataset"""
        if self._is_local:
            return self._local_data[idx]
        else:
            return dict(self.hf_dataset[idx])
    
    def format_prompt(self, item: Dict[str, Any]) -> str:
        """Format prompt with brief answer instruction"""
        question = self.extract_question(item, self._question_key)
        return f"Answer briefly.\n\n{question}"
    
    def get_response(self, item: Dict[str, Any]) -> str:
        """Get answer for training"""
        return self.extract_answer(item, self._answer_key)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._get_item_at(idx)
        
        # Extract and process image
        image = self.extract_image(item, self._image_key, self.config.max_image_size)
        if image is None:
            image = ImageUtils.create_placeholder(self.config.placeholder_size)
        
        # Process for training
        prompt = self.format_prompt(item)
        response = self.get_response(item)
        
        return self.process_for_training(image, prompt, response)
    
    def get_raw_item(self, idx: int) -> Dict[str, Any]:
        """Get raw item for evaluation"""
        item = self._get_item_at(idx)
        return {
            'question': self.extract_question(item, self._question_key),
            'answers': self.extract_all_answers(item, self._answer_key),
            'answer': self.extract_answer(item, self._answer_key),
            'image_path': item.get('image_path'),
        }


@DatasetRegistry.register("docvqa")
class DocVQADataset(BenchmarkDataset):
    """Document Visual Question Answering dataset"""
    
    def __init__(
        self,
        source: Union[str, Path] = "docvqa",
        processor = None,
        config: Optional[DatasetConfig] = None
    ):
        # If source is "docvqa", load from HF; otherwise treat as path
        super().__init__(source, processor, config, benchmark_type="docvqa")


@DatasetRegistry.register("chartqa")
class ChartQADataset(BenchmarkDataset):
    """Chart Question Answering dataset"""
    
    def __init__(
        self,
        source: Union[str, Path] = "chartqa",
        processor = None,
        config: Optional[DatasetConfig] = None
    ):
        super().__init__(source, processor, config, benchmark_type="chartqa")


@DatasetRegistry.register("ocrbench")
class OCRBenchDataset(BenchmarkDataset):
    """OCR Benchmark dataset"""
    
    def __init__(
        self,
        source: Union[str, Path] = "ocrbench",
        processor = None,
        config: Optional[DatasetConfig] = None
    ):
        super().__init__(source, processor, config, benchmark_type="ocrbench")


class BenchmarkDPODataset(BaseDPODataset, BenchmarkMixin):
    """
    Benchmark dataset formatted for DPO training.
    
    Creates preference pairs by:
    - Chosen = correct answer from dataset
    - Rejected = random answer from other samples
    
    Supports both HuggingFace datasets and local JSON files.
    """
    
    def __init__(
        self,
        benchmark_name_or_path: Union[str, Path],
        processor,
        config: Optional[DatasetConfig] = None,
        benchmark_type: Optional[str] = None
    ):
        super().__init__(processor, config)
        self.benchmark_name = benchmark_type or self._infer_benchmark_type(benchmark_name_or_path)
        self.hf_dataset = None
        self._local_data: Optional[List[Dict[str, Any]]] = None
        self._is_local = False
        self._all_answers: List[str] = []
        self.rng = random.Random(config.subset_seed if config else 42)
        self.load_data(benchmark_name_or_path)
    
    def _infer_benchmark_type(self, source: Union[str, Path]) -> str:
        """Infer benchmark type from filename or source"""
        source_str = str(source).lower()
        
        if "docvqa" in source_str:
            return "docvqa"
        elif "chartqa" in source_str:
            return "chartqa"
        elif "ocrbench" in source_str:
            return "ocrbench"
        
        return "generic"
    
    def load_data(self, source: Union[str, Path]) -> None:
        """Load benchmark and collect all answers for rejection sampling"""
        import json
        
        source_path = Path(source)
        
        if source_path.exists() and source_path.suffix == '.json':
            self._load_from_json(source_path)
        else:
            self._load_from_huggingface(str(source))
    
    def _load_from_json(self, json_path: Path) -> None:
        """Load from local JSON file"""
        import json
        
        logger.info(f"Loading benchmark for DPO from local JSON: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self._local_data = json.load(f)
        
        self._is_local = True
        
        benchmark_config = BENCHMARK_CONFIGS.get(self.benchmark_name, {})
        self._image_key = benchmark_config.get('local_image_key', 'image_path')
        self._question_key = benchmark_config.get('local_question_key', 'question')
        self._answer_key = benchmark_config.get('local_answer_key', 'answer')
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(self._local_data):
            self._local_data = self._local_data[:self.config.max_samples]
        
        # Collect all answers for rejection sampling
        for item in self._local_data:
            answer = self.extract_answer(item, self._answer_key)
            if answer and answer != "Unknown":
                self._all_answers.append(answer)
        
        self._length = len(self._local_data)
        logger.info(f"Loaded {self._length} samples, {len(set(self._all_answers))} unique answers")
    
    def _load_from_huggingface(self, source: str) -> None:
        """Load from HuggingFace datasets"""
        from datasets import load_dataset
        
        benchmark_config = BENCHMARK_CONFIGS.get(source, {})
        
        hf_name = benchmark_config.get('hf_name', source)
        split = benchmark_config.get('split', 'test')
        
        logger.info(f"Loading {source} for DPO from HuggingFace...")
        
        self.hf_dataset = load_dataset(
            hf_name,
            split=split,
            trust_remote_code=True
        )
        
        self._is_local = False
        self._image_key = benchmark_config.get('image_key', 'image')
        self._question_key = benchmark_config.get('question_key', 'question')
        self._answer_key = benchmark_config.get('answer_key', 'answers')
        
        # Apply limits
        if self.config.max_samples and self.config.max_samples < len(self.hf_dataset):
            self.hf_dataset = self.hf_dataset.select(range(self.config.max_samples))
        
        # Collect all answers for rejection sampling
        for item in self.hf_dataset:
            answer = self.extract_answer(item, self._answer_key)
            if answer and answer != "Unknown":
                self._all_answers.append(answer)
        
        self._length = len(self.hf_dataset)
        logger.info(f"Loaded {self._length} samples, {len(set(self._all_answers))} unique answers")
    
    def _get_item_at(self, idx: int) -> Dict[str, Any]:
        """Get raw item at index"""
        if self._is_local:
            return self._local_data[idx]
        else:
            return dict(self.hf_dataset[idx])
    
    def get_prompt(self, item: Dict[str, Any]) -> str:
        question = self.extract_question(item, self._question_key)
        return f"Answer briefly.\n\n{question}"
    
    def get_chosen(self, item: Dict[str, Any]) -> str:
        return self.extract_answer(item, self._answer_key)
    
    def get_rejected(self, item: Dict[str, Any]) -> str:
        """Get a random different answer as rejected"""
        correct = self.get_chosen(item)
        
        # Try to find a different answer
        for _ in range(10):
            rejected = self.rng.choice(self._all_answers) if self._all_answers else "I don't know"
            if rejected != correct:
                return rejected
        
        return "I don't know"
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._get_item_at(idx)
        
        # Extract image
        image = self.extract_image(item, self._image_key, self.config.max_image_size)
        
        # Get prompt and responses
        prompt = self.get_prompt(item)
        chosen = self.get_chosen(item)
        rejected = self.get_rejected(item)
        
        return self.to_chat_format(prompt, chosen, rejected, image)


class BenchmarkDatasetIterator:
    """
    Iterator for cached benchmark datasets.
    
    Used by evaluators that work with cached dataset format
    (list of dicts with 'image_path' keys).
    """
    
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        load_image_fn,
        evaluator_name: str = "Benchmark"
    ):
        self.dataset = dataset
        self.load_image_fn = load_image_fn
        self.evaluator_name = evaluator_name
        self.skipped_no_path = 0
        self.skipped_load_failed = 0
    
    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Image.Image]]:
        """Iterate, yielding (item, image) tuples"""
        for item in self.dataset:
            image_path = item.get('image_path')
            if not image_path:
                self.skipped_no_path += 1
                continue
            
            image = self.load_image_fn(image_path)
            if image is None:
                self.skipped_load_failed += 1
                continue
            
            yield item, image
    
    def get_skip_counts(self) -> Tuple[int, int]:
        return self.skipped_no_path, self.skipped_load_failed
    
    def get_total_skipped(self) -> int:
        return self.skipped_no_path + self.skipped_load_failed
    
    def log_skip_summary(self) -> None:
        total = self.get_total_skipped()
        if total > 0:
            logger.warning(
                f"{self.evaluator_name}: Skipped {total} samples "
                f"({self.skipped_no_path} no path, {self.skipped_load_failed} load failed)"
            )


# Factory function
def create_benchmark_dataset(
    source: Union[str, Path],
    processor,
    max_samples: int = None,
    for_dpo: bool = False,
    benchmark_type: Optional[str] = None
) -> Union[BenchmarkDataset, BenchmarkDPODataset]:
    """
    Factory function to create benchmark dataset.
    
    Args:
        source: Benchmark name (docvqa, chartqa, ocrbench) or path to local JSON
        processor: VLM processor
        max_samples: Maximum samples
        for_dpo: If True, return DPO-formatted dataset
        benchmark_type: Override inferred benchmark type ("docvqa", "chartqa", "ocrbench")
        
    Returns:
        Appropriate benchmark dataset instance
        
    Examples:
        # From HuggingFace
        dataset = create_benchmark_dataset("docvqa", processor)
        
        # From local JSON
        dataset = create_benchmark_dataset(
            "/path/to/echo840_OCRBench_test.json",
            processor,
            benchmark_type="ocrbench"
        )
        
        # Auto-infer type from filename
        dataset = create_benchmark_dataset(
            "/path/to/HuggingFaceM4_ChartQA_test.json",
            processor
        )
    """
    config = DatasetConfig(max_samples=max_samples)
    
    if for_dpo:
        return BenchmarkDPODataset(source, processor, config, benchmark_type=benchmark_type)
    return BenchmarkDataset(source, processor, config, benchmark_type=benchmark_type)
