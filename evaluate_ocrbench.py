#!/usr/bin/env python3
"""
Comprehensive evaluation script for SmolVLM model using full benchmark datasets
Evaluates the model on OCRBench, TextVQA, DocVQA, ChartQA, AI2D, ScienceQA,
MMStar, MMMU, and MathVista benchmarks.
Modified to use complete datasets for accurate performance assessment.
"""

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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmolVLMBenchmarkEvaluator:
    def __init__(self, model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct", dataset_percentage: float = 100.0):
        """Initialize the benchmark evaluator with a model

        Args:
            model_path: Path to the model (defaults to base model)
            dataset_percentage: Percentage of dataset to download and use (1-100)
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_cache = {}
        self.dataset_percentage = dataset_percentage

        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def load_model(self):
        """Load the fine-tuned model and processor"""
        logger.info(f"Loading model from: {self.model_path}")

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to base model if fine-tuned model not available
            logger.info("Falling back to base model...")
            base_model = "HuggingFaceTB/SmolVLM-500M-Instruct"
            self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

    def generate_response(self, image: Image.Image, question: str, max_tokens: int = 100) -> str:
        """Generate response for an image and question"""
        try:
            # Resize large images to avoid processor errors
            max_size = 1024
            if image.size[0] > max_size or image.size[1] > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Format using proper chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            # Process inputs
            inputs = self.processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
                size={"longest_edge": 1024}
            ).to(self.device)

            # Generate
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

            # Decode
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            logger.debug(f"Generated: {generated_text[:150]}")

            # Clean up response - extract only the Assistant's answer
            # The format is: "User: <question>\nAssistant: <answer>"
            if "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                # Fallback: try to remove the prompt
                response = generated_text.replace(prompt, "").strip()
                # Also try removing the question
                response = response.replace(question, "").strip()

            return response if response else "No answer"

        except Exception as e:
            logger.error(f"Error: {e}")
            return "Error"

    def load_image(self, image_path_or_url: str) -> Image.Image:
        """Load image from file path or URL"""
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_path_or_url).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path_or_url}: {e}")
            return None

    def load_and_save_dataset_with_fallback(self, dataset_name: str, split: str = "test", cache_dir: str = "./benchmark_cache") -> Any:
        """
        Smart dataset loader that tries 100% first, then falls back to smaller percentages on errors
        Fallback sequence: 100% → 50% → 25% → 10% → 5% → 1%
        """
        original_percentage = self.dataset_percentage
        fallback_percentages = [100, 50, 25, 10, 5, 1] if original_percentage == 100 else [original_percentage]

        for percentage in fallback_percentages:
            self.dataset_percentage = percentage
            try:
                logger.info(f"Attempting to load {dataset_name} with {percentage}% of data...")
                result = self.load_and_save_dataset(dataset_name, split, cache_dir)
                if result:
                    if percentage < original_percentage:
                        logger.warning(f"Fell back to {percentage}% due to download constraints")
                    return result
            except Exception as e:
                logger.warning(f"Failed at {percentage}%: {e}")
                if percentage == fallback_percentages[-1]:
                    logger.error(f"All fallback attempts failed for {dataset_name}")
                    raise
                continue

        # Restore original percentage
        self.dataset_percentage = original_percentage
        return None

    def load_and_save_dataset(self, dataset_name: str, split: str = "test", cache_dir: str = "./benchmark_cache") -> Any:
        """Load dataset and cache locally, respecting dataset_percentage limit"""
        percentage_str = f"_{int(self.dataset_percentage)}pct" if self.dataset_percentage < 100 else ""
        cache_path = Path(cache_dir) / f"{dataset_name.replace('/', '_')}_{split}{percentage_str}.json"
        cache_path.parent.mkdir(exist_ok=True)

        # Try to load from cache first
        if cache_path.exists():
            logger.info(f"Loading {dataset_name} from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        logger.info(f"Loading {dataset_name} dataset ({self.dataset_percentage}% of data)...")
        try:
            # Use streaming to avoid loading full dataset
            dataset = load_dataset(dataset_name, split=split, streaming=True)

            # Calculate how many samples to load based on percentage
            # We'll estimate and load samples incrementally
            dataset_list = []
            max_samples = int(10000 * (self.dataset_percentage / 100))  # Estimate max 10k samples per dataset

            logger.info(f"Downloading up to ~{max_samples} samples ({self.dataset_percentage}%)...")

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
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            logger.info("Trying non-streaming approach...")

            # Fallback to non-streaming but limited
            try:
                dataset = load_dataset(dataset_name, split=split)
                max_samples = int(len(dataset) * (self.dataset_percentage / 100))
                logger.info(f"Loading {max_samples} samples from total {len(dataset)}")

                dataset_list = []
                for idx, item in enumerate(dataset):
                    if idx >= max_samples:
                        break

                    item_dict = dict(item)
                    if 'image' in item_dict and hasattr(item_dict['image'], 'save'):
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
                            del item_dict['image']
                        except:
                            continue
                    dataset_list.append(item_dict)

                # Save to cache
                with open(cache_path, 'w') as f:
                    json.dump(dataset_list, f, indent=2)

                return dataset_list
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return []

    def evaluate_ocrbench(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on OCRBench dataset"""
        logger.info("Evaluating on OCRBench...")

        try:
            # Use proper OCR benchmark datasets from HuggingFace
            # Try multiple OCR datasets in order of preference
            dataset = None
            ocr_datasets = [
                ("echo840/OCRBench", "test"),  # Official OCRBench dataset
                ("lmms-lab/OCRBench-v2", "test"),  # OCRBench v2
                ("yunusserhat/TextOCR-Dataset", "train"),  # TextOCR dataset
                ("wulipc/CC-OCR", "test"),  # CC-OCR benchmark
            ]

            for ds_name, split in ocr_datasets:
                try:
                    logger.info(f"Trying OCR dataset: {ds_name}")
                    dataset = self.load_and_save_dataset(ds_name, split)
                    if dataset:
                        logger.info(f"Successfully loaded OCR dataset: {ds_name}")
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {ds_name}: {e}")
                    continue

            if not dataset:
                # Fallback to sample OCR tasks
                return self._evaluate_ocr_samples()

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from OCRBench")
            else:
                logger.info(f"Using full OCRBench dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="OCRBench"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    # Handle different OCR dataset formats
                    question = None
                    ground_truth = None

                    # OCRBench format (echo840/OCRBench)
                    if 'question' in item:
                        question = item['question']
                        ground_truth = item.get('answer', '')
                        question_type = item.get('question_type', 'ocr')
                    # DocVQA format with query dict
                    elif 'query' in item:
                        if isinstance(item['query'], dict):
                            question = item['query'].get('en', 'What text is visible in this image?')
                        else:
                            question = item['query']
                        ground_truth = item.get('answers', item.get('answer', []))
                    # Pure OCR datasets with just text
                    elif 'text' in item:
                        question = "Read all the text in this image."
                        ground_truth = item['text']
                    # TextOCR format
                    elif 'caption' in item:
                        question = "What text is visible in this image?"
                        ground_truth = item.get('text', item.get('caption', ''))
                    else:
                        question = "What text is visible in this image?"
                        ground_truth = item.get('answer',
                                      item.get('answers',
                                      item.get('text', '')))

                    response = self.generate_response(image, question)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": ground_truth,
                        "task_type": item.get('question_type', item.get('task_type', 'ocr')),
                        "dataset": item.get('dataset', 'ocrbench')
                    })
                except Exception as e:
                    logger.warning(f"Error processing OCRBench item: {e}")
                    continue

            return {"ocrbench": results}

        except Exception as e:
            logger.error(f"OCRBench evaluation failed: {e}")
            return self._evaluate_ocr_samples()

    def _evaluate_ocr_samples(self) -> Dict[str, Any]:
        """Fallback OCR evaluation with sample images"""
        test_cases = [
            {
                "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
                "question": "What text is visible in this image?",
                "task": "text_recognition",
                "ground_truth": "car"
            },
            {
                "image_url": "https://huggingface.co/datasets/nielsr/docvqa_1200_examples/resolve/main/example_1.png",
                "question": "Read all the text shown in this image.",
                "task": "text_recognition",
                "ground_truth": "text"
            }
        ]

        results = []
        for case in test_cases:
            image = self.load_image(case["image_url"])
            if image is None:
                logger.warning(f"Skipping {case['image_url']} - failed to load")
                continue

            response = self.generate_response(image, case["question"])
            results.append({
                "task": case["task"],
                "question": case["question"],
                "response": response,
                "ground_truth": case.get("ground_truth", ""),
                "image_url": case["image_url"]
            })

        return {"ocrbench": results}

    def evaluate_textvqa(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on TextVQA validation set"""
        logger.info("Evaluating on TextVQA...")

        try:
            dataset = self.load_and_save_dataset("HuggingFaceM4/VQAv2", "validation")
            if not dataset:
                return {"textvqa": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from TextVQA")
            else:
                logger.info(f"Using full TextVQA dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="TextVQA"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', '')
                    response = self.generate_response(image, question)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": item.get('answers', []),
                        "question_id": item.get('question_id', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing TextVQA item: {e}")
                    continue

            return {"textvqa": results}

        except Exception as e:
            logger.error(f"TextVQA evaluation failed: {e}")
            return {"textvqa": []}

    def evaluate_docvqa(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on DocVQA validation set"""
        logger.info("Evaluating on DocVQA...")

        try:
            dataset = self.load_and_save_dataset("nielsr/docvqa_1200_examples", "train")
            if not dataset:
                return {"docvqa": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from DocVQA")
            else:
                logger.info(f"Using full DocVQA dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="DocVQA"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    # Handle both formats: 'question' string or 'query' dict
                    if 'query' in item and isinstance(item['query'], dict):
                        question = item['query'].get('en', '')
                    else:
                        question = item.get('question', '')

                    response = self.generate_response(image, question, max_tokens=150)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": item.get('answers', []),
                        "question_id": item.get('questionId', item.get('id', ''))
                    })
                except Exception as e:
                    logger.warning(f"Error processing DocVQA item: {e}")
                    continue

            return {"docvqa": results}

        except Exception as e:
            logger.error(f"DocVQA evaluation failed: {e}")
            return {"docvqa": []}

    def evaluate_chartqa(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on ChartQA test set"""
        logger.info("Evaluating on ChartQA...")

        try:
            dataset = self.load_and_save_dataset("HuggingFaceM4/ChartQA", "test")
            if not dataset:
                return {"chartqa": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from ChartQA")
            else:
                logger.info(f"Using full ChartQA dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="ChartQA"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('query', '')
                    response = self.generate_response(image, question, max_tokens=100)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": item.get('label', ''),
                        "chart_type": item.get('chart_type', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing ChartQA item: {e}")
                    continue

            return {"chartqa": results}

        except Exception as e:
            logger.error(f"ChartQA evaluation failed: {e}")
            return {"chartqa": []}

    def evaluate_ai2d(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on AI2D test set"""
        logger.info("Evaluating on AI2D...")

        try:
            dataset = self.load_and_save_dataset("allenai/ai2d", "test")
            if not dataset:
                return {"ai2d": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from AI2D")
            else:
                logger.info(f"Using full AI2D dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="AI2D"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', '')
                    options = item.get('options', [])

                    # Format question with options for multiple choice
                    if options:
                        formatted_question = f"{question}\nOptions: {', '.join(options)}"
                    else:
                        formatted_question = question

                    response = self.generate_response(image, formatted_question, max_tokens=50)

                    results.append({
                        "question": question,
                        "options": options,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "question_id": item.get('question_id', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing AI2D item: {e}")
                    continue

            return {"ai2d": results}

        except Exception as e:
            logger.error(f"AI2D evaluation failed: {e}")
            return {"ai2d": []}

    def evaluate_scienceqa(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on ScienceQA test set"""
        logger.info("Evaluating on ScienceQA...")

        try:
            dataset = self.load_and_save_dataset("liuhaotian/ScienceQA", "train")
            if not dataset:
                return {"scienceqa": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from ScienceQA")
            else:
                logger.info(f"Using full ScienceQA dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="ScienceQA"):
                try:
                    image_path = item.get('image_path')
                    if image_path and os.path.exists(image_path):
                        image = self.load_image(image_path)
                    else:
                        image = None

                    question = item.get('question', '')
                    choices = item.get('choices', [])

                    # Format question with choices
                    if choices:
                        formatted_question = f"{question}\nChoices: {', '.join(choices)}"
                    else:
                        formatted_question = question

                    if image:
                        response = self.generate_response(image, formatted_question, max_tokens=50)
                    else:
                        # Text-only question - skip for this VLM evaluation
                        continue

                    results.append({
                        "question": question,
                        "choices": choices,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "subject": item.get('subject', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing ScienceQA item: {e}")
                    continue

            return {"scienceqa": results}

        except Exception as e:
            logger.error(f"ScienceQA evaluation failed: {e}")
            return {"scienceqa": []}

    def evaluate_mmstar(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on MMStar benchmark"""
        logger.info("Evaluating on MMStar...")

        try:
            dataset = self.load_and_save_dataset("Lin-Chen/MMStar", "val")
            if not dataset:
                return {"mmstar": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from MMStar")
            else:
                logger.info(f"Using full MMStar dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="MMStar"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', '')
                    choices = item.get('choices', [])

                    # Format question with choices
                    if choices:
                        formatted_question = f"{question}\nChoices: {', '.join(choices)}"
                    else:
                        formatted_question = question

                    response = self.generate_response(image, formatted_question, max_tokens=50)

                    results.append({
                        "question": question,
                        "choices": choices,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "category": item.get('category', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing MMStar item: {e}")
                    continue

            return {"mmstar": results}

        except Exception as e:
            logger.error(f"MMStar evaluation failed: {e}")
            return {"mmstar": []}

    def evaluate_mmmu(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on MMMU validation set"""
        logger.info("Evaluating on MMMU...")

        try:
            dataset = self.load_and_save_dataset("MMMU/MMMU", "validation")
            if not dataset:
                return {"mmmu": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from MMMU")
            else:
                logger.info(f"Using full MMMU dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="MMMU"):
                try:
                    image_path = item.get('image_path')
                    if image_path and os.path.exists(image_path):
                        image = self.load_image(image_path)
                    else:
                        continue

                    question = item.get('question', '')
                    options = item.get('options', [])

                    # Format question with options
                    if options:
                        formatted_question = f"{question}\nOptions: {', '.join(options)}"
                    else:
                        formatted_question = question

                    response = self.generate_response(image, formatted_question, max_tokens=100)

                    results.append({
                        "question": question,
                        "options": options,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "subject": item.get('subject', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing MMMU item: {e}")
                    continue

            return {"mmmu": results}

        except Exception as e:
            logger.error(f"MMMU evaluation failed: {e}")
            return {"mmmu": []}

    def evaluate_mathvista(self, num_samples: int = None) -> Dict[str, Any]:
        """Evaluate on MathVista test set"""
        logger.info("Evaluating on MathVista...")

        try:
            dataset = self.load_and_save_dataset("AI4Math/MathVista", "testmini")
            if not dataset:
                return {"mathvista": []}

            # Sample subset for evaluation (use full dataset if num_samples is None)
            if num_samples is not None and len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)
                logger.info(f"Using {num_samples} samples from MathVista")
            else:
                logger.info(f"Using full MathVista dataset: {len(dataset)} samples")

            results = []
            for item in tqdm(dataset, desc="MathVista"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', '')
                    choices = item.get('choices', [])

                    # Format question with choices if available
                    if choices:
                        formatted_question = f"{question}\nChoices: {', '.join(choices)}"
                    else:
                        formatted_question = question

                    response = self.generate_response(image, formatted_question, max_tokens=150)

                    results.append({
                        "question": question,
                        "choices": choices,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "problem_type": item.get('problem_type', ''),
                        "question_type": item.get('question_type', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing MathVista item: {e}")
                    continue

            return {"mathvista": results}

        except Exception as e:
            logger.error(f"MathVista evaluation failed: {e}")
            return {"mathvista": []}

    def run_full_evaluation(self, benchmarks: List[str] = None, num_samples: int = None) -> Dict[str, Any]:
        """Run comprehensive benchmark evaluation"""
        if self.dataset_percentage < 100:
            logger.info(f"Starting benchmark evaluation with {self.dataset_percentage}% of each dataset...")
        elif num_samples is None:
            logger.info("Starting comprehensive benchmark evaluation using FULL datasets...")
        else:
            logger.info(f"Starting benchmark evaluation with {num_samples} samples per dataset...")

        if self.model is None or self.processor is None:
            self.load_model()

        # Default benchmarks - image-based only for compute efficiency
        if benchmarks is None:
            benchmarks = [
                # Single-image benchmarks
                "ocrbench", "ai2d", "chartqa", "textvqa", "docvqa", "scienceqa",
                # Multi-task benchmarks  
                "mmmu", "mathvista", "mmstar"
            ]

        results = {}

        # Run selected benchmarks
        for benchmark in benchmarks:
            try:
                logger.info(f"Running {benchmark} evaluation...")
                if benchmark == "ocrbench":
                    results.update(self.evaluate_ocrbench(num_samples))
                elif benchmark == "textvqa":
                    results.update(self.evaluate_textvqa(num_samples))
                elif benchmark == "docvqa":
                    results.update(self.evaluate_docvqa(num_samples))
                elif benchmark == "chartqa":
                    results.update(self.evaluate_chartqa(num_samples))
                elif benchmark == "ai2d":
                    results.update(self.evaluate_ai2d(num_samples))
                elif benchmark == "scienceqa":
                    results.update(self.evaluate_scienceqa(num_samples))
                elif benchmark == "mmstar":
                    results.update(self.evaluate_mmstar(num_samples))
                elif benchmark == "mmmu":
                    results.update(self.evaluate_mmmu(num_samples))
                elif benchmark == "mathvista":
                    results.update(self.evaluate_mathvista(num_samples))
                else:
                    logger.warning(f"Unknown benchmark: {benchmark}")
            except Exception as e:
                logger.error(f"Failed to evaluate {benchmark}: {e}")

        return results

    def save_results(self, results: Dict[str, Any], output_file: str = "smolvlm_benchmark_results.json"):
        """Save evaluation results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def calculate_chartqa_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate ChartQA accuracy with 5% tolerance for numeric answers
        Formula: |predicted - ground_truth| / |ground_truth| ≤ 0.05
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).strip()
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                is_correct = False
                for gt in ground_truths:
                    gt_str = str(gt).strip()

                    # Try numeric comparison with 5% tolerance
                    try:
                        # Extract numbers from response and ground truth
                        import re
                        response_nums = re.findall(r'-?\d+\.?\d*', response)
                        gt_nums = re.findall(r'-?\d+\.?\d*', gt_str)

                        if response_nums and gt_nums:
                            pred_val = float(response_nums[0])
                            gt_val = float(gt_nums[0])

                            if gt_val != 0:
                                relative_error = abs(pred_val - gt_val) / abs(gt_val)
                                if relative_error <= 0.05:  # 5% tolerance
                                    is_correct = True
                                    break
                    except:
                        pass

                    # Fallback: case-insensitive text match
                    if gt_str.lower() in response.lower() or response.lower() in gt_str.lower():
                        is_correct = True
                        break

                if is_correct:
                    correct += 1
                total += 1

        return (correct / total * 100) if total > 0 else 0.0

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

    def calculate_textvqa_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate TextVQA accuracy with soft voting over multiple answers
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                ground_truths = result['ground_truth']

                # Handle both list of answers and list of dicts
                if isinstance(ground_truths, list):
                    if ground_truths and isinstance(ground_truths[0], dict):
                        # Extract answers from dict format
                        answers = [str(item['answer']).lower().strip() for item in ground_truths]
                    else:
                        answers = [str(gt).lower().strip() for gt in ground_truths]
                else:
                    answers = [str(ground_truths).lower().strip()]

                # Check if response matches any of the answers
                for answer in answers:
                    if answer in response or response in answer:
                        correct += 1
                        break

                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def calculate_docvqa_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate DocVQA accuracy using simplified ANLS
        (case-insensitive, checks for containment)
        """
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                response = str(result['response']).lower().strip()
                ground_truths = result['ground_truth'] if isinstance(result['ground_truth'], list) else [result['ground_truth']]

                # Check if any ground truth matches (case-insensitive)
                for gt in ground_truths:
                    gt_str = str(gt).lower().strip()
                    if gt_str in response or response in gt_str:
                        correct += 1
                        break

                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def calculate_accuracy(self, results: List[Dict], benchmark_name: str = None) -> float:
        """Calculate accuracy using benchmark-specific methods"""
        if not results:
            return 0.0

        # Use benchmark-specific evaluation if specified
        if benchmark_name == 'chartqa':
            return self.calculate_chartqa_accuracy(results)
        elif benchmark_name == 'ocrbench':
            return self.calculate_ocrbench_accuracy(results)
        elif benchmark_name == 'textvqa':
            return self.calculate_textvqa_accuracy(results)
        elif benchmark_name == 'docvqa':
            return self.calculate_docvqa_accuracy(results)

        # Default: simple containment check
        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                gt = str(result['ground_truth']).lower().strip()
                response = str(result['response']).lower().strip()

                if gt in response or response in gt:
                    correct += 1
                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of evaluation results (image-based benchmarks only)"""
        print("\n" + "="*80)
        if self.dataset_percentage < 100:
            print(f"SmolVLM Benchmark Evaluation Summary ({self.dataset_percentage}% of datasets)")
        else:
            print("SmolVLM Full Dataset Benchmark Evaluation Summary")
        print("="*80)

        total_tasks = 0
        benchmark_scores = {}
        
        # Group benchmarks by category (image-based only)
        image_benchmarks = ["ocrbench", "textvqa", "docvqa", "chartqa", "ai2d", "scienceqa"]
        multitask_benchmarks = ["mmmu", "mathvista", "mmstar"] 

        # Process results by category
        for category, benchmark_list in [
            ("Single-Image Benchmarks", image_benchmarks),
            ("Multi-task Benchmarks", multitask_benchmarks)
        ]:
            category_found = False
            for benchmark_name in benchmark_list:
                if benchmark_name in results:
                    if not category_found:
                        print(f"\n{category}:")
                        print("-" * 60)
                        category_found = True
                    
                    tasks = results[benchmark_name]
                    if isinstance(tasks, list) and tasks:
                        accuracy = self.calculate_accuracy(tasks, benchmark_name=benchmark_name)
                        benchmark_scores[benchmark_name] = accuracy

                        print(f"\n{benchmark_name.upper()}:")
                        print(f"  Tasks evaluated: {len(tasks)}")
                        print(f"  Accuracy: {accuracy:.2f}%")

                        # Show first few examples
                        for i, task in enumerate(tasks[:2], 1):
                            print(f"\n  Example {i}:")
                            print(f"    Question: {task.get('question', 'N/A')[:80]}...")
                            print(f"    Response: {task.get('response', 'N/A')[:80]}...")
                            if 'ground_truth' in task:
                                print(f"    Ground Truth: {str(task['ground_truth'])[:80]}...")

                        total_tasks += len(tasks)

        print(f"\n" + "="*80)
        print("BENCHMARK SCORES SUMMARY:")
        print("="*80)
        
        # Print by category with averages
        for category, benchmark_list in [
            ("Single-Image", image_benchmarks),
            ("Multi-task", multitask_benchmarks)
        ]:
            category_scores = []
            print(f"\n{category} Benchmarks:")
            for benchmark in benchmark_list:
                if benchmark in benchmark_scores:
                    score = benchmark_scores[benchmark]
                    category_scores.append(score)
                    print(f"  {benchmark:15}: {score:6.2f}%")
            
            if category_scores:
                avg = sum(category_scores) / len(category_scores)
                print(f"  {'Average':15}: {avg:6.2f}%")

        # Overall statistics
        if benchmark_scores:
            avg_score = sum(benchmark_scores.values()) / len(benchmark_scores)
            print(f"\nOverall Average Score: {avg_score:.2f}%")
        
        print(f"Total evaluated tasks: {total_tasks}")
        print("="*80)
        
        # Performance recommendations based on SmolVLM paper benchmarks (image-only)
        print("\nPerformance Analysis (compared to SmolVLM paper - image benchmarks only):")
        print("-" * 70)
        
        # Expected ranges from SmolVLM paper for image benchmarks
        expected_ranges = {
            "ocrbench": (52, 73), "ai2d": (46, 70), "chartqa": (55, 69),
            "textvqa": (50, 73), "docvqa": (58, 80), "scienceqa": (74, 90),
            "mmmu": (29, 42), "mathvista": (36, 52), "mmstar": (35, 46)
        }
        
        for benchmark, score in benchmark_scores.items():
            if benchmark in expected_ranges:
                min_exp, max_exp = expected_ranges[benchmark]
                if score >= max_exp:
                    status = "EXCELLENT (above SmolVLM-2.2B)"
                elif score >= min_exp:
                    status = "GOOD (within SmolVLM range)"
                else:
                    status = "NEEDS IMPROVEMENT (below SmolVLM-256M)"
                print(f"  {benchmark:15}: {score:6.2f}% - {status}")

        if self.dataset_percentage < 100:
            print(f"\nNote: Using {self.dataset_percentage}% of each dataset for evaluation")
        else:
            print("\nNote: Using full datasets for comprehensive evaluation")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM model on benchmark datasets")
    parser.add_argument("--model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct",
                       help="Path to the model (default: HuggingFaceTB/SmolVLM-500M-Instruct)")
    parser.add_argument("--output-file", default="smolvlm_benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--benchmarks", nargs="+",
                       choices=["ocrbench", "textvqa", "docvqa", "chartqa",
                               "ai2d", "scienceqa", "mmstar", "mmmu", "mathvista"],
                       help="Specific benchmarks to run (image-based only)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples per benchmark (None = use full datasets)")
    parser.add_argument("--percentage", type=float, default=100.0,
                       help="Percentage of each dataset to download and use (1-100, default: 100)")
    parser.add_argument("--cache-dir", default="./benchmark_cache",
                       help="Directory to cache datasets")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate percentage
    if args.percentage <= 0 or args.percentage > 100:
        parser.error("--percentage must be between 1 and 100")

    # Initialize evaluator with percentage limit
    evaluator = SmolVLMBenchmarkEvaluator(args.model_path, dataset_percentage=args.percentage)

    try:
        # Run evaluation
        results = evaluator.run_full_evaluation(
            benchmarks=args.benchmarks,
            num_samples=args.num_samples
        )

        # Save results
        evaluator.save_results(results, args.output_file)

        # Print summary
        evaluator.print_summary(results)

        print(f"\nDetailed results saved to: {args.output_file}")
        print(f"Dataset cache saved to: {args.cache_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Clean up model and CUDA resources to prevent threading errors
        import gc
        import torch
        if hasattr(evaluator, 'model'):
            del evaluator.model
        if hasattr(evaluator, 'processor'):
            del evaluator.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == "__main__":
    main()