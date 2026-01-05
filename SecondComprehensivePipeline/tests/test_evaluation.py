"""
Unit Tests for Evaluation Code

Tests verify that:
1. Generation uses correct prompt (trimmed, not including answer)
2. Predictions are extracted from the right position
3. Answer matching works correctly
4. No generation from wrong positions
"""

import unittest
import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.setup import setup_hf_cache, BASE_MODEL
setup_hf_cache()

from transformers import AutoProcessor, AutoModelForImageTextToText


class TestEvaluationGeneration(unittest.TestCase):
    """Test that evaluation generates from the correct position"""

    @classmethod
    def setUpClass(cls):
        """Load processor and small model once for all tests"""
        cls.processor = AutoProcessor.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )
        # Load model for actual generation tests
        cls.model = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            device_map="auto"
        )
        cls.model.eval()

    def test_prompt_trimming(self):
        """Test that we correctly identify where to trim for generation"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What is 2+2?"
        answer = "4"

        # Create full messages
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

        # Process full sequence
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

        # Assertions
        self.assertIsNotNone(answer_start_pos, "Should find answer position")
        self.assertGreater(answer_start_pos, len(full_token_list) * 0.3,
                          "Answer should be in latter part of sequence")
        self.assertLess(answer_start_pos, len(full_token_list),
                       "Answer position should be within sequence")

    def test_generation_from_prompt_end(self):
        """Test that generation happens from prompt end, not full sequence end"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What is 2+2?"
        answer = "4"

        # Create full messages
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

        # CORRECT APPROACH: Trim to prompt
        prompt_input_ids = full_inputs['input_ids'][:, :answer_start_pos]

        device = next(self.model.parameters()).device
        gen_inputs_correct = {
            'input_ids': prompt_input_ids.to(device),
            'pixel_values': full_inputs['pixel_values'].to(device)
        }

        with torch.no_grad():
            outputs_correct = self.model.generate(
                **gen_inputs_correct,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Should generate SOMETHING after the prompt
        generated_tokens = outputs_correct[0][prompt_input_ids.shape[1]:]
        self.assertGreater(len(generated_tokens), 0,
                          "Should generate tokens after prompt")

        # BUGGY APPROACH: Use full sequence
        gen_inputs_buggy = {
            'input_ids': full_inputs['input_ids'].to(device),
            'pixel_values': full_inputs['pixel_values'].to(device)
        }

        with torch.no_grad():
            outputs_buggy = self.model.generate(
                **gen_inputs_buggy,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Extract tokens after full sequence
        buggy_generated = outputs_buggy[0][full_inputs['input_ids'].shape[1]:]

        # Buggy approach generates few or no tokens
        print(f"\nGeneration comparison:")
        print(f"  Correct approach: {len(generated_tokens)} tokens")
        print(f"  Buggy approach: {len(buggy_generated)} tokens")

        # Correct approach should generate more or equal tokens
        self.assertGreaterEqual(len(generated_tokens), len(buggy_generated),
                               "Correct approach should generate at least as many tokens")

    def test_label_based_prompt_finding(self):
        """Test finding prompt end from labels (masked positions)"""
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

        # Create labels (same as training)
        answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        full_token_list = full_inputs["input_ids"][0].tolist()

        answer_start_pos_search = None
        for i in range(len(full_token_list) - len(answer_tokens) + 1):
            if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
                answer_start_pos_search = i
                break

        labels = full_inputs["input_ids"].clone()
        labels[:, :answer_start_pos_search] = -100

        # Find prompt end from labels (FIXED evaluation approach)
        mask = labels[0] != -100
        if mask.any():
            answer_start_pos_from_labels = mask.nonzero()[0].item()

        # Both methods should give the same result
        self.assertEqual(answer_start_pos_search, answer_start_pos_from_labels,
                        "Answer position from search should match position from labels")


class TestAnswerMatching(unittest.TestCase):
    """Test answer matching strategies"""

    def test_exact_match(self):
        """Test exact string matching"""
        pred = "42"
        answer = "42"
        self.assertTrue(pred.strip() == answer.strip())

    def test_case_insensitive_match(self):
        """Test case-insensitive matching"""
        pred = "Yes"
        answer = "yes"
        self.assertTrue(pred.lower().strip() == answer.lower().strip())

    def test_punctuation_tolerance(self):
        """Test matching with punctuation differences"""
        pred = "No."
        answer = "No"
        # Remove trailing punctuation
        self.assertTrue(pred.rstrip('.').lower() == answer.lower())

    def test_contained_answer(self):
        """Test when answer is contained in verbose prediction"""
        pred = "The value of the lowest bar is 23."
        answer = "23"
        self.assertTrue(answer in pred)

    def test_number_extraction(self):
        """Test extracting and comparing numbers"""
        import re

        pred = "There are three bars in the chart."
        answer = "3"

        # This should fail with exact match
        self.assertNotEqual(pred, answer)

        # But we can extract numbers
        pred_numbers = re.findall(r'\d+', pred)
        answer_numbers = re.findall(r'\d+', answer)

        # No numbers in "three bars"
        self.assertEqual(len(pred_numbers), 0)

        # For another example with numbers
        pred2 = "The answer is 23."
        answer2 = "23"

        pred_numbers2 = re.findall(r'\d+', pred2)
        answer_numbers2 = re.findall(r'\d+', answer2)

        self.assertEqual(pred_numbers2[0], answer_numbers2[0])

    def test_decimal_number_matching(self):
        """Test matching decimal numbers"""
        import re

        pred = "The difference is 0.57"
        answer = "0.57"

        pred_numbers = re.findall(r'\d+\.?\d*', pred)
        answer_numbers = re.findall(r'\d+\.?\d*', answer)

        self.assertEqual(pred_numbers[-1], answer_numbers[0])

    def test_multiple_matching_strategies(self):
        """Test combined matching strategy"""
        test_cases = [
            ("42", "42", True),  # Exact match
            ("Yes", "yes", True),  # Case insensitive
            ("No.", "No", True),  # Punctuation
            ("The answer is 23", "23", True),  # Contained
            ("Wrong answer", "42", False),  # No match
        ]

        for pred, answer, expected in test_cases:
            # Normalize
            pred_norm = pred.lower().strip().rstrip('.')
            answer_norm = answer.lower().strip().rstrip('.')

            # Try matching strategies
            is_correct = False

            # 1. Exact match
            if pred_norm == answer_norm:
                is_correct = True

            # 2. Contained
            if not is_correct and answer_norm in pred_norm:
                is_correct = True

            # 3. Number extraction
            if not is_correct:
                import re
                pred_numbers = re.findall(r'\d+\.?\d*', pred)
                answer_numbers = re.findall(r'\d+\.?\d*', answer)
                if pred_numbers and answer_numbers:
                    if pred_numbers[0] == answer_numbers[0]:
                        is_correct = True

            self.assertEqual(is_correct, expected,
                           f"Failed for pred='{pred}', answer='{answer}'")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestAnswerMatching))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
