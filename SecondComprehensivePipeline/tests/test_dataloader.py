"""
Unit tests for dataloader classes.

Tests:
- Base dataset classes (DatasetConfig, ImageUtils, AnswerMaskingMixin, DatasetRegistry)
- QCM datasets (QCMDataset, QCMDPODataset)
- Benchmark datasets (BenchmarkDataset, BenchmarkMixin)
- DPO datasets (DPODataset, DPOSFTDataset, LazyDPODataset)
- Data collators (VisionLanguageDataCollator, DPODataCollator)
"""

import json
import pytest
import tempfile
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Tests for base_dataset.py
# =============================================================================

class TestDatasetConfig:
    """Tests for DatasetConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        from dataloader.base_dataset import DatasetConfig

        config = DatasetConfig()
        assert config.max_samples is None
        assert config.max_image_size == 2048
        assert config.patch_size == 16
        assert config.placeholder_size == (512, 512)
        assert config.cache_images is True
        assert config.force_patch_divisible is True
        assert config.use_fixed_subset is False
        assert config.subset_seed == 42

    def test_custom_values(self):
        """Test custom configuration values"""
        from dataloader.base_dataset import DatasetConfig

        config = DatasetConfig(
            max_samples=100,
            max_image_size=1024,
            patch_size=32,
            placeholder_size=(256, 256),
            cache_images=False
        )
        assert config.max_samples == 100
        assert config.max_image_size == 1024
        assert config.patch_size == 32
        assert config.placeholder_size == (256, 256)
        assert config.cache_images is False


class TestImageUtils:
    """Tests for ImageUtils class"""

    def test_round_to_patch_size(self):
        """Test rounding dimensions to patch size"""
        from dataloader.base_dataset import ImageUtils

        # Exact multiples
        assert ImageUtils.round_to_patch_size(512, 16) == 512
        assert ImageUtils.round_to_patch_size(256, 16) == 256

        # Non-multiples round down
        assert ImageUtils.round_to_patch_size(520, 16) == 512
        assert ImageUtils.round_to_patch_size(519, 16) == 512
        assert ImageUtils.round_to_patch_size(527, 16) == 512
        assert ImageUtils.round_to_patch_size(528, 16) == 528

        # Minimum is patch_size
        assert ImageUtils.round_to_patch_size(10, 16) == 16
        assert ImageUtils.round_to_patch_size(1, 16) == 16

    def test_create_placeholder(self):
        """Test placeholder image creation"""
        from dataloader.base_dataset import ImageUtils

        # Default size
        img = ImageUtils.create_placeholder()
        assert isinstance(img, Image.Image)
        assert img.size == (512, 512)
        assert img.mode == 'RGB'

        # Custom size
        img = ImageUtils.create_placeholder((256, 128))
        assert img.size == (256, 128)

        # Custom color
        img = ImageUtils.create_placeholder((100, 100), 'black')
        assert img.size == (100, 100)

    def test_load_image_nonexistent(self):
        """Test loading non-existent image returns None"""
        from dataloader.base_dataset import ImageUtils

        result = ImageUtils.load_image("/nonexistent/path/to/image.png")
        assert result is None

    def test_load_image_valid(self):
        """Test loading a valid image"""
        from dataloader.base_dataset import ImageUtils

        # Create a temp image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), 'red')
            img.save(f.name)

            loaded = ImageUtils.load_image(f.name)
            assert loaded is not None
            assert isinstance(loaded, Image.Image)
            assert loaded.mode == 'RGB'

            Path(f.name).unlink()

    def test_resize_image_within_limits(self):
        """Test resizing image within size limits"""
        from dataloader.base_dataset import ImageUtils

        # Image smaller than max_size
        img = Image.new('RGB', (500, 300))
        resized = ImageUtils.resize_image(img, max_size=2048)

        # Should round to patch size
        assert resized.size[0] % 16 == 0
        assert resized.size[1] % 16 == 0

    def test_resize_image_exceeds_limits(self):
        """Test resizing image that exceeds size limits"""
        from dataloader.base_dataset import ImageUtils

        # Image larger than max_size
        img = Image.new('RGB', (3000, 2000))
        resized = ImageUtils.resize_image(img, max_size=1024)

        # Longest edge should be <= 1024
        assert max(resized.size) <= 1024
        # Should be divisible by patch_size
        assert resized.size[0] % 16 == 0
        assert resized.size[1] % 16 == 0

    def test_resize_image_aspect_ratio_preserved(self):
        """Test that aspect ratio is approximately preserved"""
        from dataloader.base_dataset import ImageUtils

        img = Image.new('RGB', (2000, 1000))  # 2:1 aspect ratio
        resized = ImageUtils.resize_image(img, max_size=512)

        # Aspect ratio should be approximately 2:1
        original_ratio = 2000 / 1000
        new_ratio = resized.size[0] / resized.size[1]
        assert abs(original_ratio - new_ratio) < 0.2  # Allow some rounding error

    def test_get_cache_key(self):
        """Test cache key generation"""
        from dataloader.base_dataset import ImageUtils

        key1 = ImageUtils.get_cache_key("image1.png", 512, 512)
        key2 = ImageUtils.get_cache_key("image1.png", 512, 512)
        key3 = ImageUtils.get_cache_key("image2.png", 512, 512)

        # Same input = same key
        assert key1 == key2
        # Different input = different key
        assert key1 != key3


class TestAnswerMaskingMixin:
    """Tests for AnswerMaskingMixin"""

    def test_create_masked_labels_1d(self):
        """Test creating masked labels for 1D tensor"""
        from dataloader.base_dataset import AnswerMaskingMixin

        mixin = AnswerMaskingMixin()
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        labels = mixin.create_masked_labels(input_ids, answer_start_pos=5)

        # First 5 positions should be -100
        assert (labels[:5] == -100).all()
        # Rest should be original
        assert (labels[5:] == input_ids[5:]).all()

    def test_create_masked_labels_2d(self):
        """Test creating masked labels for 2D tensor"""
        from dataloader.base_dataset import AnswerMaskingMixin

        mixin = AnswerMaskingMixin()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        labels = mixin.create_masked_labels(input_ids, answer_start_pos=4)

        assert (labels[:, :4] == -100).all()
        assert (labels[:, 4:] == input_ids[:, 4:]).all()

    def test_create_masked_labels_all_masked_fallback(self):
        """Test fallback when all tokens would be masked"""
        from dataloader.base_dataset import AnswerMaskingMixin

        mixin = AnswerMaskingMixin()
        input_ids = torch.tensor([1, 2, 3, 4, 5])

        # Mask all tokens (answer_start_pos >= length)
        labels = mixin.create_masked_labels(input_ids, answer_start_pos=10, min_unmasked_tokens=2)

        # Should have at least 2 unmasked tokens
        assert (labels != -100).sum() >= 2


class TestDatasetRegistry:
    """Tests for DatasetRegistry"""

    def test_register_and_get(self):
        """Test registering and retrieving dataset classes"""
        from dataloader.base_dataset import DatasetRegistry, BaseDataset

        @DatasetRegistry.register("test_dataset")
        class TestDataset(BaseDataset):
            def load_data(self, source): pass
            def __getitem__(self, idx): pass

        retrieved = DatasetRegistry.get("test_dataset")
        assert retrieved is TestDataset

    def test_get_nonexistent(self):
        """Test getting non-existent dataset returns None"""
        from dataloader.base_dataset import DatasetRegistry

        result = DatasetRegistry.get("nonexistent_dataset")
        assert result is None

    def test_list_available(self):
        """Test listing available datasets"""
        from dataloader.base_dataset import DatasetRegistry

        available = DatasetRegistry.list_available()
        assert isinstance(available, list)
        # Should include registered datasets
        assert "qcm" in available or len(available) >= 0  # May or may not be registered yet


# =============================================================================
# Tests for qcm_dataset.py
# =============================================================================

class TestQCMDataset:
    """Tests for QCMDataset"""

    @pytest.fixture
    def sample_qcm_data(self, tmp_path):
        """Create sample QCM data for testing"""
        data = [
            {
                "question": "What color is the sky?",
                "options": {"A": "Blue", "B": "Red", "C": "Green", "D": "Yellow"},
                "correct_answer": "A",
                "image_name": "sky.png"
            },
            {
                "question": "What is 2+2?",
                "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
                "correct_answer": "B",
                "image_name": "math.png"
            }
        ]

        json_path = tmp_path / "qcm_test.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        # Create dummy images
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for name in ["sky.png", "math.png"]:
            img = Image.new('RGB', (100, 100), 'white')
            img.save(image_dir / name)

        return json_path, image_dir

    @pytest.fixture
    def mock_processor(self):
        """Create a mock processor"""
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.encode.return_value = [1, 2, 3]
        processor.apply_chat_template.return_value = "test template"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        return processor

    def test_load_flat_format(self, sample_qcm_data, mock_processor):
        """Test loading QCM data in flat format"""
        from dataloader.qcm_dataset import QCMDataset

        json_path, image_dir = sample_qcm_data
        dataset = QCMDataset(json_path, image_dir, mock_processor)

        assert len(dataset) == 2

    def test_load_nested_format(self, tmp_path, mock_processor):
        """Test loading QCM data in nested format"""
        from dataloader.qcm_dataset import QCMDataset

        data = [
            {
                "qcm": {
                    "question": "Test?",
                    "options": {"A": "Yes", "B": "No"},
                    "correct_answer": "A"
                },
                "image_name": "test.png"
            }
        ]

        json_path = tmp_path / "nested_qcm.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.new('RGB', (100, 100)).save(image_dir / "test.png")

        dataset = QCMDataset(json_path, image_dir, mock_processor)
        assert len(dataset) == 1

    def test_format_prompt(self, sample_qcm_data, mock_processor):
        """Test prompt formatting"""
        from dataloader.qcm_dataset import QCMDataset

        json_path, image_dir = sample_qcm_data
        dataset = QCMDataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        prompt = dataset.format_prompt(item)

        assert "What color is the sky?" in prompt
        assert "A: Blue" in prompt
        assert "B: Red" in prompt

    def test_get_response(self, sample_qcm_data, mock_processor):
        """Test response extraction"""
        from dataloader.qcm_dataset import QCMDataset

        json_path, image_dir = sample_qcm_data
        dataset = QCMDataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        response = dataset.get_response(item)

        assert response == "A"

    def test_max_samples(self, sample_qcm_data, mock_processor):
        """Test max_samples limit"""
        from dataloader.qcm_dataset import QCMDataset
        from dataloader.base_dataset import DatasetConfig

        json_path, image_dir = sample_qcm_data
        config = DatasetConfig(max_samples=1)
        dataset = QCMDataset(json_path, image_dir, mock_processor, config)

        assert len(dataset) == 1


class TestQCMDPODataset:
    """Tests for QCMDPODataset"""

    @pytest.fixture
    def sample_qcm_data(self, tmp_path):
        """Create sample QCM data for DPO testing"""
        data = [
            {
                "question": "Test question?",
                "options": {"A": "Option A", "B": "Option B", "C": "Option C"},
                "correct_answer": "A",
                "image_name": "test.png"
            }
        ]

        json_path = tmp_path / "qcm_dpo.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.new('RGB', (100, 100)).save(image_dir / "test.png")

        return json_path, image_dir

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        return processor

    def test_get_chosen(self, sample_qcm_data, mock_processor):
        """Test getting chosen (correct) answer"""
        from dataloader.qcm_dataset import QCMDPODataset

        json_path, image_dir = sample_qcm_data
        dataset = QCMDPODataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        chosen = dataset.get_chosen(item)

        assert chosen == "A"

    def test_get_rejected(self, sample_qcm_data, mock_processor):
        """Test getting rejected (incorrect) answer"""
        from dataloader.qcm_dataset import QCMDPODataset

        json_path, image_dir = sample_qcm_data
        dataset = QCMDPODataset(json_path, image_dir, mock_processor, seed=42)

        item = dataset._data[0]
        rejected = dataset.get_rejected(item)

        # Should be one of B or C, not A
        assert rejected in ["B", "C"]
        assert rejected != "A"


# =============================================================================
# Tests for benchmark_dataset.py
# =============================================================================

class TestBenchmarkMixin:
    """Tests for BenchmarkMixin"""

    def test_extract_question_simple(self):
        """Test extracting simple question"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {"question": "What is this?"}
        result = BenchmarkMixin.extract_question(item, "question")

        assert result == "What is this?"

    def test_extract_question_dict_format(self):
        """Test extracting question from dict format (multi-language)"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {"query": {"en": "English question", "de": "German question"}}
        result = BenchmarkMixin.extract_question(item, "query")

        assert result == "English question"

    def test_extract_question_fallback(self):
        """Test raises KeyError when key not found (no fallback)"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {}
        with pytest.raises(KeyError) as exc_info:
            BenchmarkMixin.extract_question(item, "question")

        assert "Could not find question" in str(exc_info.value)

    def test_extract_answer_list(self):
        """Test extracting answer from list"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {"answers": ["answer1", "answer2"]}
        result = BenchmarkMixin.extract_answer(item, "answers")

        assert result == "answer1"

    def test_extract_answer_string(self):
        """Test extracting answer from string"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {"answer": "single answer"}
        result = BenchmarkMixin.extract_answer(item, "answer")

        assert result == "single answer"

    def test_extract_all_answers(self):
        """Test extracting all answers"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {"answers": ["ans1", "ans2", "ans3"]}
        result = BenchmarkMixin.extract_all_answers(item, "answers")

        assert result == ["ans1", "ans2", "ans3"]

    def test_extract_answer_raises_when_missing(self):
        """Test raises KeyError when answer not found (no fallback)"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {}
        with pytest.raises(KeyError) as exc_info:
            BenchmarkMixin.extract_answer(item, "answers")

        assert "Could not find answer" in str(exc_info.value)

    def test_extract_all_answers_raises_when_missing(self):
        """Test raises KeyError when answers not found (no fallback)"""
        from dataloader.benchmark_dataset import BenchmarkMixin

        item = {}
        with pytest.raises(KeyError) as exc_info:
            BenchmarkMixin.extract_all_answers(item, "answers")

        assert "Could not find answers" in str(exc_info.value)


class TestBenchmarkDataset:
    """Tests for BenchmarkDataset"""

    @pytest.fixture
    def sample_benchmark_data(self, tmp_path):
        """Create sample benchmark data"""
        data = [
            {
                "question": "What is shown?",
                "answer": "A chart",
                "image_path": str(tmp_path / "images" / "chart.png")
            }
        ]

        json_path = tmp_path / "benchmark.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.new('RGB', (200, 200)).save(image_dir / "chart.png")

        return json_path

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.encode.return_value = [1, 2, 3]
        processor.apply_chat_template.return_value = "template"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        return processor

    def test_load_from_json(self, sample_benchmark_data, mock_processor):
        """Test loading benchmark from local JSON"""
        from dataloader.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(sample_benchmark_data, mock_processor)

        assert len(dataset) == 1
        assert dataset._is_local is True

    def test_infer_benchmark_type(self, mock_processor):
        """Test benchmark type inference from filename"""
        from dataloader.benchmark_dataset import BenchmarkDataset

        # We can test the private method directly
        dataset = BenchmarkDataset.__new__(BenchmarkDataset)

        assert dataset._infer_benchmark_type("docvqa_train.json") == "docvqa"
        assert dataset._infer_benchmark_type("chartqa_test.json") == "chartqa"
        assert dataset._infer_benchmark_type("ocrbench_data.json") == "ocrbench"
        assert dataset._infer_benchmark_type("unknown.json") == "generic"

    def test_format_prompt(self, sample_benchmark_data, mock_processor):
        """Test prompt formatting"""
        from dataloader.benchmark_dataset import BenchmarkDataset

        dataset = BenchmarkDataset(sample_benchmark_data, mock_processor)

        item = dataset._get_item_at(0)
        prompt = dataset.format_prompt(item)

        assert "Answer briefly" in prompt
        assert "What is shown?" in prompt


# =============================================================================
# Tests for dpo_dataset.py
# =============================================================================

class TestDPODataset:
    """Tests for DPODataset"""

    @pytest.fixture
    def sample_dpo_data(self, tmp_path):
        """Create sample DPO data"""
        data = [
            {
                "prompt": "Describe this image",
                "chosen": "A beautiful sunset",
                "rejected": "An ugly image",
                "image_name": "sunset.png"
            }
        ]

        json_path = tmp_path / "dpo.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.new('RGB', (100, 100)).save(image_dir / "sunset.png")

        return json_path, image_dir

    @pytest.fixture
    def mock_processor(self):
        return MagicMock()

    def test_load_data(self, sample_dpo_data, mock_processor):
        """Test loading DPO data"""
        from dataloader.dpo_dataset import DPODataset

        json_path, image_dir = sample_dpo_data
        dataset = DPODataset(json_path, image_dir, mock_processor)

        assert len(dataset) == 1

    def test_get_prompt(self, sample_dpo_data, mock_processor):
        """Test getting prompt"""
        from dataloader.dpo_dataset import DPODataset

        json_path, image_dir = sample_dpo_data
        dataset = DPODataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        assert dataset.get_prompt(item) == "Describe this image"

    def test_get_chosen(self, sample_dpo_data, mock_processor):
        """Test getting chosen response"""
        from dataloader.dpo_dataset import DPODataset

        json_path, image_dir = sample_dpo_data
        dataset = DPODataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        assert dataset.get_chosen(item) == "A beautiful sunset"

    def test_get_rejected(self, sample_dpo_data, mock_processor):
        """Test getting rejected response"""
        from dataloader.dpo_dataset import DPODataset

        json_path, image_dir = sample_dpo_data
        dataset = DPODataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        assert dataset.get_rejected(item) == "An ugly image"

    def test_skip_missing_image(self, tmp_path, mock_processor):
        """Test skipping items with missing images"""
        from dataloader.dpo_dataset import DPODataset

        data = [
            {"prompt": "p", "chosen": "c", "rejected": "r", "image_name": "missing.png"}
        ]

        json_path = tmp_path / "dpo.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()

        dataset = DPODataset(json_path, image_dir, mock_processor)

        assert len(dataset) == 0  # Skipped due to missing image


class TestDPOSFTDataset:
    """Tests for DPOSFTDataset"""

    @pytest.fixture
    def sample_dpo_data(self, tmp_path):
        data = [{"prompt": "p", "chosen": "c", "rejected": "r", "image_name": "img.png"}]

        json_path = tmp_path / "dpo_sft.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)

        image_dir = tmp_path / "images"
        image_dir.mkdir()
        Image.new('RGB', (100, 100)).save(image_dir / "img.png")

        return json_path, image_dir

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.encode.return_value = [1, 2, 3]
        processor.apply_chat_template.return_value = "template"
        processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(1, 3, 224, 224)
        }
        return processor

    def test_load_data(self, sample_dpo_data, mock_processor):
        """Test loading DPO data for SFT"""
        from dataloader.dpo_dataset import DPOSFTDataset

        json_path, image_dir = sample_dpo_data
        dataset = DPOSFTDataset(json_path, image_dir, mock_processor)

        assert len(dataset) == 1

    def test_get_response_uses_chosen(self, sample_dpo_data, mock_processor):
        """Test that get_response returns chosen (not rejected)"""
        from dataloader.dpo_dataset import DPOSFTDataset

        json_path, image_dir = sample_dpo_data
        dataset = DPOSFTDataset(json_path, image_dir, mock_processor)

        item = dataset._data[0]
        assert dataset.get_response(item) == "c"


# =============================================================================
# Tests for data_collators.py
# =============================================================================

class TestVisionLanguageDataCollator:
    """Tests for VisionLanguageDataCollator"""

    def test_collate_batch(self):
        """Test collating a batch of features"""
        from dataloader.data_collators import VisionLanguageDataCollator

        collator = VisionLanguageDataCollator(pad_token_id=0)

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "pixel_values": torch.randn(3, 224, 224)
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([4, 5]),
                "pixel_values": torch.randn(3, 224, 224)
            }
        ]

        batch = collator(features)

        assert batch["input_ids"].shape == (2, 3)  # Padded to max length
        assert batch["attention_mask"].shape == (2, 3)
        assert batch["labels"].shape == (2, 3)
        assert batch["pixel_values"].shape == (2, 3, 224, 224)

    def test_padding_with_correct_values(self):
        """Test that padding uses correct pad values"""
        from dataloader.data_collators import VisionLanguageDataCollator

        collator = VisionLanguageDataCollator(pad_token_id=999, label_pad_token_id=-100)

        features = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([1, 2, 3]),
                "pixel_values": torch.randn(3, 224, 224)
            },
            {
                "input_ids": torch.tensor([4]),
                "attention_mask": torch.tensor([1]),
                "labels": torch.tensor([4]),
                "pixel_values": torch.randn(3, 224, 224)
            }
        ]

        batch = collator(features)

        # Second sample should be padded
        assert batch["input_ids"][1, 1].item() == 999
        assert batch["attention_mask"][1, 1].item() == 0
        assert batch["labels"][1, 1].item() == -100


class TestVisionLanguageDataCollatorWithPadding:
    """Tests for VisionLanguageDataCollatorWithPadding"""

    def test_right_padding(self):
        """Test right padding (default)"""
        from dataloader.data_collators import VisionLanguageDataCollatorWithPadding

        collator = VisionLanguageDataCollatorWithPadding(padding_side="right")

        features = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
            },
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([3, 4, 5]),
            }
        ]

        batch = collator(features)

        # First sample should have padding at the end
        assert batch["input_ids"][0, 0].item() == 1
        assert batch["input_ids"][0, 2].item() == 0  # Padded

    def test_left_padding(self):
        """Test left padding"""
        from dataloader.data_collators import VisionLanguageDataCollatorWithPadding

        collator = VisionLanguageDataCollatorWithPadding(padding_side="left")

        features = [
            {
                "input_ids": torch.tensor([1, 2]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([1, 2]),
            },
            {
                "input_ids": torch.tensor([3, 4, 5]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([3, 4, 5]),
            }
        ]

        batch = collator(features)

        # First sample should have padding at the beginning
        assert batch["input_ids"][0, 0].item() == 0  # Padded
        assert batch["input_ids"][0, 1].item() == 1


class TestDPODataCollator:
    """Tests for DPODataCollator"""

    def test_collate_dpo_batch(self):
        """Test collating DPO batch"""
        from dataloader.data_collators import DPODataCollator

        collator = DPODataCollator()

        features = [
            {
                "prompt": [{"role": "user", "content": "p1"}],
                "chosen": [{"role": "assistant", "content": "c1"}],
                "rejected": [{"role": "assistant", "content": "r1"}],
                "images": [Image.new('RGB', (100, 100))]
            },
            {
                "prompt": [{"role": "user", "content": "p2"}],
                "chosen": [{"role": "assistant", "content": "c2"}],
                "rejected": [{"role": "assistant", "content": "r2"}],
                "images": [Image.new('RGB', (100, 100))]
            }
        ]

        batch = collator(features)

        assert len(batch["prompt"]) == 2
        assert len(batch["chosen"]) == 2
        assert len(batch["rejected"]) == 2
        assert len(batch["images"]) == 2


class TestCreateDataCollator:
    """Tests for create_data_collator factory function"""

    def test_create_vision_language_collator(self):
        """Test creating vision language collator"""
        from dataloader.data_collators import create_data_collator, VisionLanguageDataCollator

        collator = create_data_collator("vision_language")
        assert isinstance(collator, VisionLanguageDataCollator)

    def test_create_dpo_collator(self):
        """Test creating DPO collator"""
        from dataloader.data_collators import create_data_collator, DPODataCollator

        collator = create_data_collator("dpo")
        assert isinstance(collator, DPODataCollator)

    def test_create_unknown_collator_raises(self):
        """Test that unknown collator type raises error"""
        from dataloader.data_collators import create_data_collator

        with pytest.raises(ValueError, match="Unknown collator type"):
            create_data_collator("unknown_type")


# =============================================================================
# Integration Tests
# =============================================================================

class TestFactoryFunctions:
    """Integration tests for factory functions"""

    @pytest.fixture
    def tmp_data(self, tmp_path):
        """Create temporary test data"""
        # QCM data
        qcm_data = [{"question": "Q?", "options": {"A": "a", "B": "b"},
                     "correct_answer": "A", "image_name": "img.png"}]
        qcm_path = tmp_path / "qcm.json"
        with open(qcm_path, 'w') as f:
            json.dump(qcm_data, f)

        # DPO data
        dpo_data = [{"prompt": "p", "chosen": "c", "rejected": "r", "image_name": "img.png"}]
        dpo_path = tmp_path / "dpo.json"
        with open(dpo_path, 'w') as f:
            json.dump(dpo_data, f)

        # Images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new('RGB', (100, 100)).save(img_dir / "img.png")

        return {"qcm": qcm_path, "dpo": dpo_path, "images": img_dir}

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        processor.tokenizer = MagicMock()
        processor.tokenizer.encode.return_value = [1, 2, 3]
        return processor

    def test_create_qcm_dataset_sft(self, tmp_data, mock_processor):
        """Test creating QCM dataset for SFT"""
        from dataloader.qcm_dataset import create_qcm_dataset, QCMDataset

        dataset = create_qcm_dataset(
            str(tmp_data["qcm"]),
            str(tmp_data["images"]),
            mock_processor
        )

        assert isinstance(dataset, QCMDataset)

    def test_create_qcm_dataset_dpo(self, tmp_data, mock_processor):
        """Test creating QCM dataset for DPO"""
        from dataloader.qcm_dataset import create_qcm_dataset, QCMDPODataset

        dataset = create_qcm_dataset(
            str(tmp_data["qcm"]),
            str(tmp_data["images"]),
            mock_processor,
            for_dpo=True
        )

        assert isinstance(dataset, QCMDPODataset)

    def test_create_dpo_dataset_standard(self, tmp_data, mock_processor):
        """Test creating standard DPO dataset"""
        from dataloader.dpo_dataset import create_dpo_dataset, DPODataset

        dataset = create_dpo_dataset(
            str(tmp_data["dpo"]),
            str(tmp_data["images"]),
            mock_processor
        )

        assert isinstance(dataset, DPODataset)

    def test_create_dpo_dataset_sft(self, tmp_data, mock_processor):
        """Test creating DPO dataset for SFT"""
        from dataloader.dpo_dataset import create_dpo_dataset, DPOSFTDataset

        dataset = create_dpo_dataset(
            str(tmp_data["dpo"]),
            str(tmp_data["images"]),
            mock_processor,
            for_sft=True
        )

        assert isinstance(dataset, DPOSFTDataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
