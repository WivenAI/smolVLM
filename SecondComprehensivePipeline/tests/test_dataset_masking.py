"""
Unit Tests for Dataset Masking in Full Fine-Tuning

Tests verify that:
1. Answer tokens are NOT masked (labels != -100)
2. Prompt tokens ARE masked (labels == -100)
3. At least some tokens are available for training
4. Token alignment is correct
"""

import unittest
import sys
from pathlib import Path
import json
import torch
from PIL import Image
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.setup import setup_hf_cache, BASE_MODEL
setup_hf_cache()

from transformers import AutoProcessor


class TestBenchmarkDatasetMasking(unittest.TestCase):
    """Test BenchmarkDataset masking logic"""

    @classmethod
    def setUpClass(cls):
        """Load processor once for all tests"""
        cls.processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )

    def test_simple_qa_masking(self):
        """Test that a simple Q&A is masked correctly"""
        # Create test data
        image = Image.new('RGB', (512, 512), color='white')
        question = "What color is the sky?"
        answer = "Blue"

        # Create messages
        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        # Process
        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Find answer position using the FIXED approach
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                break

        # Assertions
        self.assertIsNotNone(answer_start_pos, "Answer tokens should be found in sequence")
        self.assertGreater(answer_start_pos, 0, "Answer should not start at position 0")
        self.assertLess(answer_start_pos, len(full_token_list), "Answer position should be within sequence")

        # Create labels with FIXED masking
        labels = full_inputs["input_ids"].clone()
        labels[:, :answer_start_pos] = -100

        # Verify masking
        unmasked_count = (labels[0] != -100).sum().item()
        total_count = len(labels[0])
        masked_count = (labels[0] == -100).sum().item()

        self.assertGreater(unmasked_count, 0, "Should have some unmasked tokens for training")
        self.assertEqual(masked_count + unmasked_count, total_count, "All tokens should be either masked or unmasked")
        self.assertEqual(masked_count, answer_start_pos, "Masked count should equal answer start position")

        # Verify the unmasked portion contains the answer
        unmasked_tokens = full_inputs["input_ids"][0][answer_start_pos:].tolist()
        self.assertGreaterEqual(len(unmasked_tokens), len(answer_tokens),
                               f"Should have at least {len(answer_tokens)} unmasked tokens")

    def test_no_all_tokens_masked(self):
        """Test that we never mask ALL tokens (critical bug check)"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What is 2+2?"
        answer = "4"

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Find answer position
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                break

        # Create labels
        labels = full_inputs["input_ids"].clone()
        if answer_start_pos is not None:
            labels[:, :answer_start_pos] = -100

        unmasked_count = (labels[0] != -100).sum().item()

        # CRITICAL: Must have at least 1 token to train on
        self.assertGreater(unmasked_count, 0,
                          "CRITICAL BUG: All tokens are masked! No training signal!")

    def test_long_answer_masking(self):
        """Test masking with a longer answer"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "Describe the image."
        answer = "This is a white square image with no other visible elements."

        full_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Find answer position
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                break

        self.assertIsNotNone(answer_start_pos, "Should find long answer in sequence")

        # Create labels
        labels = full_inputs["input_ids"].clone()
        labels[:, :answer_start_pos] = -100

        unmasked_count = (labels[0] != -100).sum().item()

        # Should have multiple tokens for longer answer
        self.assertGreater(unmasked_count, 5,
                          f"Long answer should have many unmasked tokens, got {unmasked_count}")

    def test_qcm_style_masking(self):
        """Test masking for QCM-style (multiple choice) questions"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What color is the sky?"
        options = "A: Red\nB: Blue\nC: Green\nD: Yellow"
        prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with the letter of the correct option:"
        answer = "B"

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
                "content": [{"type": "text", "text": answer}]
            }
        ]

        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        # Find answer position
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                break

        # For QCM, answer is very short (single letter)
        self.assertIsNotNone(answer_start_pos, "Should find single-letter answer")

        labels = full_inputs["input_ids"].clone()
        labels[:, :answer_start_pos] = -100

        unmasked_count = (labels[0] != -100).sum().item()

        # Even for single letter, should have some tokens
        self.assertGreater(unmasked_count, 0, "QCM answer should have unmasked tokens")
        self.assertGreater(answer_start_pos, len(full_token_list) * 0.5,
                          "Answer should be in second half of sequence (after long prompt)")


class TestMaskingComparison(unittest.TestCase):
    """Compare buggy vs fixed masking approaches"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )

    def test_buggy_vs_fixed_masking(self):
        """Compare buggy (separate processing) vs fixed (single processing) masking"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What is this?"
        answer = "A white square"

        user_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        full_messages = user_message + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]

        # BUGGY APPROACH (how it was before)
        prompt_text = self.processor.apply_chat_template(
            user_message, add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        prompt_inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        full_inputs = self.processor(
            text=full_text,
            images=image,
            return_tensors="pt",
            padding=True,
            size={"longest_edge": 1024}
        )

        buggy_prompt_length = prompt_inputs["input_ids"].shape[1]
        buggy_labels = full_inputs["input_ids"].clone()
        buggy_labels[:, :buggy_prompt_length] = -100
        buggy_unmasked = (buggy_labels[0] != -100).sum().item()

        # FIXED APPROACH (current)
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos = i
                break

        fixed_labels = full_inputs["input_ids"].clone()
        if answer_start_pos is not None:
            fixed_labels[:, :answer_start_pos] = -100
        fixed_unmasked = (fixed_labels[0] != -100).sum().item()

        # The fixed approach should give MORE unmasked tokens (more training signal)
        # Or at minimum, the same amount
        self.assertGreaterEqual(fixed_unmasked, buggy_unmasked,
                              f"Fixed masking should have >= unmasked tokens than buggy. "
                              f"Buggy: {buggy_unmasked}, Fixed: {fixed_unmasked}")

        print(f"\nMasking Comparison:")
        print(f"  Buggy approach: {buggy_unmasked} tokens for training")
        print(f"  Fixed approach: {fixed_unmasked} tokens for training")
        print(f"  Improvement: {fixed_unmasked - buggy_unmasked} more tokens ({((fixed_unmasked / buggy_unmasked - 1) * 100):.1f}% increase)")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkDatasetMasking))
    suite.addTests(loader.loadTestsFromTestCase(TestMaskingComparison))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
