"""
Base Evaluator - Shared functionality for all evaluators
"""
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

# Set HuggingFace cache before imports (must be before transformers/peft)
from config.setup import setup_hf_cache, get_hf_cache_dir, BASE_MODEL
setup_hf_cache()

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for all evaluators with shared functionality"""

    def __init__(self, cache_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent / "datasets" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir = get_hf_cache_dir()

    def _cleanup_model(self):
        """Free GPU memory from previous model"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_path: str):
        """Load model from path (handles both full models and LoRA adapters)"""
        self._cleanup_model()  # Free memory from previous model
        logger.info(f"Loading model from: {model_path}")

        model_path = Path(model_path)
        is_adapter = (model_path / "adapter_config.json").exists()

        if is_adapter:
            logger.info("Detected LoRA adapter, loading base model first...")
            base_model = BASE_MODEL

            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=self.hf_cache_dir
            )

            base_model_obj = AutoModelForImageTextToText.from_pretrained(
                base_model,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.hf_cache_dir
            )

            self.model = PeftModel.from_pretrained(base_model_obj, model_path)
            logger.info("LoRA adapter loaded successfully")
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                cache_dir=self.hf_cache_dir
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.hf_cache_dir
            )
            logger.info("Full model loaded successfully")

        self.model.eval()

    def load_base_model(self, base_model: str = None):
        """Load the base model without fine-tuning"""
        if base_model is None:
            base_model = BASE_MODEL
        self._cleanup_model()  # Free memory from previous model
        logger.info(f"Loading base model: {base_model}")

        self.processor = AutoProcessor.from_pretrained(
            base_model,
            trust_remote_code=True,
            cache_dir=self.hf_cache_dir
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir=self.hf_cache_dir
        )
        self.model.eval()
        logger.info("Base model loaded successfully")

    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load image from path"""
        try:
            path = Path(image_path)
            if path.exists():
                return Image.open(path).convert('RGB')
            else:
                logger.warning(f"Image not found: {image_path}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def generate_response(self, image: Image.Image, question: str, max_new_tokens: int = 256) -> str:
        """Generate response for image+question"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Match the training prompt format (with "Answer briefly." prefix)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # Get device from model's first parameter (handles device_map="auto")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode only new tokens
        response = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

    def load_cached_dataset(self, dataset_name: str, split: str = "test", max_samples: int = None) -> List[Dict]:
        """Load dataset from cache or download and cache"""
        cache_name = f"{dataset_name.replace('/', '_')}_{split}"
        cache_path = self.cache_dir / f"{cache_name}.json"

        # Try cache first
        if cache_path.exists():
            logger.info(f"Loading {dataset_name} from cache...")
            with open(cache_path, 'r') as f:
                data = json.load(f)

            # Check if cache has enough samples
            if max_samples and len(data) < max_samples:
                logger.warning(f"Cache has {len(data)} samples but {max_samples} requested. Re-downloading...")
            else:
                if max_samples:
                    data = data[:max_samples]
                return data

        # Download and cache
        logger.info(f"Downloading {dataset_name}...")
        dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,
            trust_remote_code=True,
            cache_dir=self.hf_cache_dir
        )

        dataset_list = []
        target_samples = max_samples or 2000

        for idx, item in enumerate(tqdm(dataset, desc=f"Downloading {dataset_name}")):
            if idx >= target_samples:
                break

            item_dict = dict(item)

            # Save image to cache if present
            if 'image' in item_dict and hasattr(item_dict['image'], 'save'):
                img_path = self.cache_dir / f"{cache_name}_{idx}.jpg"
                try:
                    image = item_dict['image']
                    if image.mode in ('RGBA', 'LA', 'P'):
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P' and 'transparency' in image.info:
                            image = image.convert('RGBA')
                        if image.mode in ('RGBA', 'LA'):
                            rgb_image.paste(image, mask=image.split()[-1])
                        else:
                            rgb_image.paste(image)
                        image = rgb_image
                    image.save(img_path, 'JPEG')
                    item_dict['image_path'] = str(img_path)
                    del item_dict['image']
                except Exception as e:
                    logger.warning(f"Failed to save image {idx}: {e}")
                    continue

            dataset_list.append(item_dict)

        # Save cache
        with open(cache_path, 'w') as f:
            json.dump(dataset_list, f, indent=2)

        logger.info(f"Cached {len(dataset_list)} samples from {dataset_name}")
        return dataset_list

    @abstractmethod
    def evaluate(self, model_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Run evaluation and return results"""
        pass

    @abstractmethod
    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate accuracy from results"""
        pass

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")
