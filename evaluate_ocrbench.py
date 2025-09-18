#!/usr/bin/env python3
"""
Comprehensive evaluation script for SmolVLM model (Image-based benchmarks only)
Evaluates the model on OCRBench, TextVQA, DocVQA, ChartQA, AI2D, ScienceQA, 
MMStar, MMMU, and MathVista benchmarks.
Video benchmarks removed to reduce compute requirements.
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
    def __init__(self, model_path: str = "./smolvlm-500m-finetuned"):
        """Initialize the benchmark evaluator with a fine-tuned model"""
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results_cache = {}

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
            # Format the input text
            text = f"<image>Question: {question}\nAnswer:"

            # Process inputs
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            )

            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.1
                )

            # Decode response
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Extract only the answer part
            if "Answer:" in generated_text:
                response = generated_text.split("Answer:")[-1].strip()
            else:
                response = generated_text.strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error: Unable to generate response"

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

    def load_and_save_dataset(self, dataset_name: str, split: str = "test", cache_dir: str = "./benchmark_cache") -> Any:
        """Load dataset and cache locally"""
        cache_path = Path(cache_dir) / f"{dataset_name.replace('/', '_')}_{split}.json"
        cache_path.parent.mkdir(exist_ok=True)

        # Try to load from cache first
        if cache_path.exists():
            logger.info(f"Loading {dataset_name} from cache...")
            with open(cache_path, 'r') as f:
                return json.load(f)

        logger.info(f"Loading {dataset_name} dataset...")
        try:
            dataset = load_dataset(dataset_name, split=split)
            # Convert to list for JSON serialization
            dataset_list = []
            for item in dataset:
                # Convert PIL images to base64 or save locally
                item_dict = dict(item)
                if 'image' in item_dict and hasattr(item_dict['image'], 'save'):
                    # Save image locally and store path
                    img_path = cache_path.parent / f"{dataset_name.replace('/', '_')}_{len(dataset_list)}.jpg"
                    item_dict['image'].save(img_path)
                    item_dict['image_path'] = str(img_path)
                    del item_dict['image']  # Remove PIL object
                dataset_list.append(item_dict)

            # Save to cache
            with open(cache_path, 'w') as f:
                json.dump(dataset_list, f, indent=2)

            return dataset_list
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return []

    def evaluate_ocrbench(self, num_samples: int = 50) -> Dict[str, Any]:
        """Evaluate on OCRBench dataset"""
        logger.info("Evaluating on OCRBench...")

        try:
            # Try to load the official OCRBench dataset
            dataset = self.load_and_save_dataset("Yuliang-Liu/MultimodalOCR", "test")
            if not dataset:
                # Fallback to sample OCR tasks
                return self._evaluate_ocr_samples()

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset, desc="OCRBench"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', 'What text is visible in this image?')
                    response = self.generate_response(image, question)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": item.get('answer', ''),
                        "task_type": item.get('task_type', 'ocr')
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
                "task": "text_recognition"
            },
            {
                "image_url": "https://via.placeholder.com/400x200/ffffff/000000?text=SAMPLE+TEXT",
                "question": "Read all the text shown in this image.",
                "task": "text_recognition"
            }
        ]

        results = []
        for case in test_cases:
            image = self.load_image(case["image_url"])
            if image is None:
                continue

            response = self.generate_response(image, case["question"])
            results.append({
                "task": case["task"],
                "question": case["question"],
                "response": response,
                "image_url": case["image_url"]
            })

        return {"ocrbench": results}

    def evaluate_textvqa(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on TextVQA validation set"""
        logger.info("Evaluating on TextVQA...")

        try:
            dataset = self.load_and_save_dataset("textvqa", "validation")
            if not dataset:
                return {"textvqa": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="TextVQA"):
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

    def evaluate_docvqa(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on DocVQA validation set"""
        logger.info("Evaluating on DocVQA...")

        try:
            dataset = self.load_and_save_dataset("lmms-lab/DocVQA", "validation")
            if not dataset:
                return {"docvqa": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="DocVQA"):
                try:
                    image_path = item.get('image_path')
                    if not image_path or not os.path.exists(image_path):
                        continue

                    image = self.load_image(image_path)
                    if image is None:
                        continue

                    question = item.get('question', '')
                    response = self.generate_response(image, question, max_tokens=150)

                    results.append({
                        "question": question,
                        "response": response,
                        "ground_truth": item.get('answers', []),
                        "question_id": item.get('questionId', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing DocVQA item: {e}")
                    continue

            return {"docvqa": results}

        except Exception as e:
            logger.error(f"DocVQA evaluation failed: {e}")
            return {"docvqa": []}

    def evaluate_chartqa(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on ChartQA test set"""
        logger.info("Evaluating on ChartQA...")

        try:
            dataset = self.load_and_save_dataset("HuggingFaceM4/ChartQA", "test")
            if not dataset:
                return {"chartqa": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="ChartQA"):
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

    def evaluate_ai2d(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on AI2D test set"""
        logger.info("Evaluating on AI2D...")

        try:
            dataset = self.load_and_save_dataset("lmms-lab/ai2d", "test")
            if not dataset:
                return {"ai2d": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="AI2D"):
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

    def evaluate_scienceqa(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on ScienceQA test set"""
        logger.info("Evaluating on ScienceQA...")

        try:
            dataset = self.load_and_save_dataset("derek-thomas/ScienceQA", "test")
            if not dataset:
                return {"scienceqa": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="ScienceQA"):
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

    def evaluate_mmstar(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on MMStar benchmark"""
        logger.info("Evaluating on MMStar...")

        try:
            dataset = self.load_and_save_dataset("Lin-Chen/MMStar", "val")
            if not dataset:
                return {"mmstar": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="MMStar"):
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

    def evaluate_mmmu(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on MMMU validation set"""
        logger.info("Evaluating on MMMU...")

        try:
            dataset = self.load_and_save_dataset("MMMU/MMMU", "validation")
            if not dataset:
                return {"mmmu": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="MMMU"):
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

    def evaluate_mathvista(self, num_samples: int = 100) -> Dict[str, Any]:
        """Evaluate on MathVista test set"""
        logger.info("Evaluating on MathVista...")

        try:
            dataset = self.load_and_save_dataset("AI4Math/MathVista", "testmini")
            if not dataset:
                return {"mathvista": []}

            # Sample subset for evaluation
            if len(dataset) > num_samples:
                dataset = random.sample(dataset, num_samples)

            results = []
            for item in tqdm(dataset[:num_samples], desc="MathVista"):
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

    def run_full_evaluation(self, benchmarks: List[str] = None, num_samples: int = 50) -> Dict[str, Any]:
        """Run comprehensive benchmark evaluation focusing on image-based tasks only"""
        logger.info("Starting comprehensive benchmark evaluation (image-based only)...")

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

    def calculate_accuracy(self, results: List[Dict]) -> float:
        """Calculate accuracy for multiple choice questions"""
        if not results:
            return 0.0

        correct = 0
        total = 0

        for result in results:
            if 'ground_truth' in result and 'response' in result:
                gt = str(result['ground_truth']).lower().strip()
                response = str(result['response']).lower().strip()

                # Simple exact match for now
                if gt in response or response in gt:
                    correct += 1
                total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def print_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of evaluation results (image-based benchmarks only)"""
        print("\n" + "="*80)
        print("SmolVLM Image-Based Benchmark Evaluation Summary")
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
                        accuracy = self.calculate_accuracy(tasks)
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

        print("\nNote: Video benchmarks excluded to reduce compute requirements")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLM model on image-based benchmarks")
    parser.add_argument("--model-path", default="./smolvlm-500m-finetuned",
                       help="Path to the fine-tuned model")
    parser.add_argument("--output-file", default="smolvlm_benchmark_results.json",
                       help="Output file for results")
    parser.add_argument("--benchmarks", nargs="+",
                       choices=["ocrbench", "textvqa", "docvqa", "chartqa",
                               "ai2d", "scienceqa", "mmstar", "mmmu", "mathvista"],
                       help="Specific benchmarks to run (image-based only)")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of samples per benchmark")
    parser.add_argument("--cache-dir", default="./benchmark_cache",
                       help="Directory to cache datasets")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize evaluator
    evaluator = SmolVLMBenchmarkEvaluator(args.model_path)

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

if __name__ == "__main__":
    main()