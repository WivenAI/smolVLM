# Set HuggingFace cache directory before importing transformers (avoids disk quota issues on clusters)
import os
_hf_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmpcache"))
os.makedirs(_hf_cache, exist_ok=True)
os.environ["HF_HOME"] = _hf_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_hf_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = _hf_cache

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests
from io import BytesIO
import json
import os
from datasets import load_dataset
from tqdm import tqdm
import argparse
from typing import List, Dict, Any
import logging
import random
import numpy as np
from pathlib import Path
from pathlib import Path
from peft import PeftModel

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ocr_evaluator():
    def init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset_name="echo840/OCRBench"

    def load_model(self):
        """Load the fine-tuned model and processor"""
        logger.info(f"Loading model from: {self.model_path}")

        # Check if this is a LoRA adapter directory
        model_path = Path(self.model_path)
        is_adapter = (model_path / "adapter_config.json").exists()

        if is_adapter:
            logger.info("Detected LoRA adapter, loading base model first...")
            base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"

            # Load processor from adapter directory (it should have processor files)
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, cache_dir=self.hf_cache_dir)

            # Load base model
            base_model_obj = AutoModelForVision2Seq.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.hf_cache_dir
            )

            # Load LoRA adapter on top
            self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
            logger.info("LoRA adapter loaded successfully on base model")
        else:
            # Load as full model
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True, cache_dir=self.hf_cache_dir)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=self.hf_cache_dir
            )
            logger.info("Full model loaded successfully")


    def load_and_save_dataset(self, dataset_name: str, split: str = "test", cache_dir: str = "./datasets/benchmark_cache") -> Any:
        """Load dataset and cache locally, respecting dataset_percentage limit"""
        cache_path = Path(cache_dir) / f"{dataset_name.replace('/', '_')}_{split}.json"
        cache_path.parent.mkdir(exist_ok=True)
        # Try to load from cache first
        if cache_path.exists():
            logger.info(f"Loading {dataset_name} from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)
        logger.info(f"Loading {dataset_name} dataset ({self.dataset_percentage}% of data)...")
        dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True, cache_dir=self.hf_cache_dir)
        max_samples = 2000
        dataset_list = []
        logger.info(f"Downloading up to ~{max_samples} samples ocrBench")
        for idx, item in enumerate(dataset):
            if idx >= max_samples:
                break
            # Convert PIL images to base64 or save locally
            item_dict = dict(item)
            if 'image' in item_dict and hasattr(item_dict['image'], 'save'):
                # Save image locally and store path
                img_path = cache_path.parent / f"{dataset_name.replace('/', '_')}_{len(dataset_list)}.jpg"
                try:
                    # Convert RGBA to RGB before saving as JPEG
                    image = item_dict['image']
                    if image.mode in ('RGBA', 'LA', 'P'):
                        # Convert to RGB (JPEG doesn't support transparency)
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P' and 'transparency' in image.info:
                            image = image.convert('RGBA')
                        rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                        image = rgb_image
                    image.save(img_path, 'JPEG')
                    item_dict['image_path'] = str(img_path)
                    del item_dict['image']  # Remove PIL object
                except Exception as e:
                    logger.warning(f"Failed to save image {idx}: {e}")
                    continue

            dataset_list.append(item_dict)
            # Progress update every 100 samples
            if (idx + 1) % 100 == 0:
                logger.info(f"Downloaded {idx + 1} samples...")
        logger.info(f"Downloaded {len(dataset_list)} samples from {dataset_name}")
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(dataset_list, f, indent=2)

        return dataset_list

    def evaluate_ocrbench(self, num_samples: int = None) -> Dict[str, Any]:
        results = []
        """Evaluate on OCRBench dataset"""
        logger.info("Evaluating on OCRBench...")
        dataset = self.load_and_save_dataset(self.dataset_name, "test")
        for item in tqdm(dataset, desc="OCRBench"):
            image_path = item.get('image_path')
            image = self.load_image(image_path)
            assert(image)
            question = item['question']
            ground_truth = item.get('answer', '')
            response = self.generate_response(image, question)

            results.append({
                "question": question,
                "response": response,
                "ground_truth": ground_truth,
                "task_type": item.get('question_type', item.get('task_type', 'ocr')),
                "dataset": item.get('dataset', 'ocrbench')
            })
        
        return {"ocrbench": results}
    
    def calculate_ocrbench_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate OCRBench accuracy - checks if ground truth is contained in prediction
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                # Check if any ground truth is contained in the response
                for gt in ground_truths:
                    gt_str = str(gt).lower().strip()
                    if gt_str in response:
                        correct += 1
                        break

                total += 1

        return (correct / total * 100) if total > 0 else 0.0
    
    def accuracy(self, results):
        return self.calculate_ocrbench_accuracy(results)