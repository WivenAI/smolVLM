# Migration Guide: Modular Dataset System

This guide shows how to migrate from the old scattered dataset code to the new unified, modular system.

## Quick Start

```python
# Import the new modular system
from datasets import (
    QCMDataset, DPODataset, BenchmarkDataset,
    DatasetConfig, VisionLanguageDataCollator,
    split_dataset, validate_dataset
)

# Create a dataset with configuration
config = DatasetConfig(max_samples=1000, max_image_size=1024)
dataset = QCMDataset("data/qcm.json", "images/", processor, config)

# Split and validate
train_ds, eval_ds = split_dataset(dataset, train_ratio=0.9)
is_valid, errors = validate_dataset(dataset)
```

---

## Example 1: QCM Dataset for SFT Training

### Before (trainer_sft.py)
```python
class QCMDataset(torch.utils.data.Dataset):
    def __init__(self, json_path: str, image_dir: str, processor):
        self.processor = processor
        self.image_dir = Path(image_dir)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.original_items = raw_data
        if raw_data and 'qcm' in raw_data[0]:
            self.data = [item['qcm'] for item in raw_data]
        else:
            self.data = raw_data
        # ... lots of duplicated image loading and processing code
```

### After (New modular system)
```python
from datasets import QCMDataset, DatasetConfig, create_qcm_dataset

# Option 1: Full control with config
config = DatasetConfig(max_samples=1000, max_image_size=1024)
dataset = QCMDataset(
    json_path="data/qcm_gemini.json",
    image_dir="images/",
    processor=processor,
    config=config
)

# Option 2: Simple factory function
dataset = create_qcm_dataset("data/qcm.json", "images/", processor, max_samples=1000)

# Option 3: Using registry
from datasets import DatasetRegistry
dataset = DatasetRegistry.create("qcm", "data/qcm.json", "images/", processor)
```

---

## Example 2: DPO Dataset

### Before (trainer_dpo.py)
```python
def prepare_dpo_dataset(self, dataset_path: str, image_dir: str, max_samples: int = None):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    image_dir = Path(image_dir).resolve()
    dpo_data = []
    skipped_missing_image = 0
    
    for item in raw_data:
        image_name = item.get('image_name', '')
        # ... lots of boilerplate for image handling
        
        dpo_data.append({
            'prompt_text': prompt,
            'chosen_text': chosen,
            'rejected_text': rejected,
            'image_path': image_path_str,
        })
    # ... more processing
```

### After (New modular system)
```python
from datasets import DPODataset, LazyDPODataset, DPOSFTDataset, DatasetConfig

# For standard DPO training:
config = DatasetConfig(max_samples=500)
dataset = DPODataset("data/dpo.json", "images/", processor, config)

# For memory-efficient lazy loading:
dataset = LazyDPODataset("data/dpo.json", "images/", processor, config)

# For SFT on chosen responses only:
dataset = DPOSFTDataset("data/dpo.json", "images/", processor, config)

# Factory function:
from datasets import create_dpo_dataset
dataset = create_dpo_dataset("data/dpo.json", "images/", processor, lazy_loading=True)
```

---

## Example 3: Benchmark Datasets

### Before (trainer_sft.py)
```python
class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, benchmark_name: str, processor, max_samples: int = None):
        if benchmark_name == "docvqa":
            self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")
        elif benchmark_name == "ocrbench":
            self.dataset = load_dataset("echo840/OCRBench", split="test")
        elif benchmark_name == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
        # ... lots of duplicated extraction logic
```

### After (New modular system)
```python
from datasets import BenchmarkDataset, DocVQADataset, ChartQADataset, OCRBenchDataset

# Generic benchmark loader:
dataset = BenchmarkDataset("docvqa", processor)

# Or specific dataset classes:
docvqa = DocVQADataset(processor)
chartqa = ChartQADataset(processor)
ocrbench = OCRBenchDataset(processor)

# Factory function:
from datasets import create_benchmark_dataset
dataset = create_benchmark_dataset("docvqa", processor, max_samples=500)
```

---

## Example 4: Data Collation

### Before (duplicated in multiple files)
```python
@dataclass
class VisionLanguageDataCollator:
    def __call__(self, features):
        pixel_values = [f.pop('pixel_values') for f in features]
        max_length = max(f['input_ids'].shape[0] for f in features)
        # ... duplicated in multiple files
```

### After (Centralized)
```python
from datasets import VisionLanguageDataCollator, create_data_collator

# Direct usage:
collator = VisionLanguageDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

# Factory:
collator = create_data_collator("vision_language", pad_token_id=0)

# Enhanced collator with more options:
from datasets import VisionLanguageDataCollatorWithPadding
collator = VisionLanguageDataCollatorWithPadding(
    pad_token_id=0,
    max_length=512,
    truncation=True
)
```

---

## Example 5: Dataset Utilities

### Before (scattered across files)
```python
# Split logic duplicated in trainer_sft.py and trainer_dpo.py
train_size = int(0.9 * dataset_size)
eval_size = dataset_size - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, eval_size],
    generator=torch.Generator().manual_seed(42)
)
```

### After (Centralized utilities)
```python
from datasets import (
    split_dataset, create_subset, validate_dataset, 
    print_dataset_info, DatasetCache
)

# Split dataset:
train_ds, eval_ds = split_dataset(dataset, train_ratio=0.9, seed=42)

# Create subset:
subset = create_subset(dataset, max_samples=100, seed=42)

# Validate dataset:
is_valid, errors = validate_dataset(dataset)
if not is_valid:
    print(f"Validation errors: {errors}")

# Print detailed info:
print_dataset_info(dataset, "Training Dataset")

# Caching:
cache = DatasetCache("./cache")
cache_key = cache.get_cache_key("data/qcm.json", "qcm", 1000)
if cache.exists(cache_key):
    cached_ds = cache.load(cache_key)
```

---

## Example 6: Using the Registry

```python
from datasets import DatasetRegistry, DatasetConfig

# List available datasets:
print(DatasetRegistry.list_available())
# Output: ['qcm', 'qcm_dpo', 'dpo', 'dpo_sft', 'benchmark', 'docvqa', 'chartqa', 'ocrbench']

# Create dataset by name:
config = DatasetConfig(max_samples=500)

# For QCM:
qcm_dataset = DatasetRegistry.create(
    "qcm",
    json_path="data/qcm.json",
    image_dir="images/",
    processor=processor,
    config=config
)

# For benchmarks:
docvqa_dataset = DatasetRegistry.create("docvqa", processor=processor, config=config)
```

---

## Complete Training Setup Example

```python
from transformers import AutoProcessor, Trainer, TrainingArguments

# Import from new modular system
from datasets import (
    QCMDataset,
    DatasetConfig,
    VisionLanguageDataCollator,
    split_dataset,
    validate_dataset,
    print_dataset_info
)

# Load processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

# Create dataset with configuration
config = DatasetConfig(
    max_samples=1000,
    max_image_size=1024,
    force_patch_divisible=True
)

dataset = QCMDataset(
    json_path="data/qcm_gemini.json",
    image_dir="images/erp_screenshots/",
    processor=processor,
    config=config
)

# Validate
is_valid, errors = validate_dataset(dataset)
if not is_valid:
    raise ValueError(f"Dataset validation failed: {errors}")

# Print info
print_dataset_info(dataset, "QCM Training Dataset")

# Split
train_dataset, eval_dataset = split_dataset(dataset, train_ratio=0.9)

# Create collator
collator = VisionLanguageDataCollator(
    pad_token_id=processor.tokenizer.pad_token_id
)

# Setup training
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)

trainer.train()
```

---

## Key Benefits

| Benefit | Description |
|---------|-------------|
| **Modularity** | Each dataset type in its own file, easy to add new types |
| **Reusability** | Common functionality in base classes, shared utilities |
| **Consistency** | All datasets follow the same interface |
| **Testability** | Small, focused classes that are easy to test |
| **Extensibility** | Registry pattern for dynamic dataset creation |
| **Maintainability** | DRY principle, single source of truth |

---

## Migration Checklist

- [ ] Replace inline `QCMDataset` with `datasets.QCMDataset`
- [ ] Replace inline `DPOSFTDataset` with `datasets.DPOSFTDataset`
- [ ] Replace inline `BenchmarkDataset` with `datasets.BenchmarkDataset`
- [ ] Replace `VisionLanguageDataCollator` with `datasets.VisionLanguageDataCollator`
- [ ] Replace manual dataset splitting with `datasets.split_dataset()`
- [ ] Replace manual image loading with `ImageUtils` methods
- [ ] Remove duplicated answer masking code (use `AnswerMaskingMixin`)
- [ ] Update imports in `__init__.py` files
- [ ] Update any tests to use new interfaces
- [ ] Add validation calls where appropriate
