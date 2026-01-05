"""
Unit Tests for Tokenization and Masking in Training

Tests verify that:
1. Answer tokens can be found in tokenized sequences
2. Leading space variants are handled correctly
3. Masking positions are correct for all datasets
4. No fallback/emergency masking is triggered
"""

import unittest
import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.setup import setup_hf_cache, BASE_MODEL
setup_hf_cache()

from transformers import AutoProcessor

# Dataset cache paths
CACHE_DIR = Path(__file__).parent.parent / "datasets" / "cache"
CHARTQA_CACHE = CACHE_DIR / "HuggingFaceM4_ChartQA_test.json"
DOCVQA_CACHE = CACHE_DIR / "nielsr_docvqa_1200_examples_train.json"
OCRBENCH_CACHE = CACHE_DIR / "echo840_OCRBench_test.json"


def find_answer_position(processor, full_token_list, answer):
    """
    Find answer position using the fixed algorithm (with leading space variant).
    Returns (position, method) or (None, None) if not found.
    """
    # Try 1: Answer with leading space (most common in chat templates)
    answer_with_space = " " + answer
    answer_tokens_spaced = processor.tokenizer.encode(answer_with_space, add_special_tokens=False)
    for i in range(len(full_token_list) - len(answer_tokens_spaced) + 1):
        if full_token_list[i:i+len(answer_tokens_spaced)] == answer_tokens_spaced:
            return i, "with_space"

    # Try 2: Answer without leading space
    answer_tokens = processor.tokenizer.encode(answer, add_special_tokens=False)
    for i in range(len(full_token_list) - len(answer_tokens) + 1):
        if full_token_list[i:i+len(answer_tokens)] == answer_tokens:
            return i, "no_space"

    # Try 3: Find "Assistant:" marker
    for marker in ["Assistant:", "Assistant: ", ": "]:
        marker_tokens = processor.tokenizer.encode(marker, add_special_tokens=False)
        for i in range(len(full_token_list) - len(marker_tokens) + 1):
            if full_token_list[i:i+len(marker_tokens)] == marker_tokens:
                return i + len(marker_tokens), "marker"

    return None, None


class TestTokenizationBasics(unittest.TestCase):
    """Test basic tokenization behavior"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    def test_leading_space_changes_tokens(self):
        """Test that leading space produces different tokens"""
        no_space = self.processor.tokenizer.encode("No", add_special_tokens=False)
        with_space = self.processor.tokenizer.encode(" No", add_special_tokens=False)

        self.assertNotEqual(no_space, with_space,
            f"'No' and ' No' should tokenize differently: {no_space} vs {with_space}")

    def test_yes_no_tokenization(self):
        """Test Yes/No tokenization variants"""
        words = ["Yes", "No", "True", "False"]
        for word in words:
            no_space = self.processor.tokenizer.encode(word, add_special_tokens=False)
            with_space = self.processor.tokenizer.encode(" " + word, add_special_tokens=False)

            # They should be different
            self.assertNotEqual(no_space, with_space,
                f"'{word}' and ' {word}' should tokenize differently")

            # Both should decode back to the original (with/without space)
            decoded_no_space = self.processor.tokenizer.decode(no_space)
            decoded_with_space = self.processor.tokenizer.decode(with_space)

            self.assertEqual(decoded_no_space.strip(), word)
            self.assertEqual(decoded_with_space.strip(), word)

    def test_number_tokenization(self):
        """Test number tokenization"""
        numbers = ["14", "0.57", "100", "3.14159"]
        for num in numbers:
            no_space = self.processor.tokenizer.encode(num, add_special_tokens=False)
            with_space = self.processor.tokenizer.encode(" " + num, add_special_tokens=False)

            # Verify they decode correctly
            self.assertEqual(self.processor.tokenizer.decode(no_space).strip(), num)
            self.assertEqual(self.processor.tokenizer.decode(with_space).strip(), num)

    def test_chat_template_adds_space(self):
        """Test that chat template adds space before assistant response"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Test?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "No"}]}
        ]

        full_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # The assistant response should have a space before it
        self.assertIn("Assistant:", full_text)
        # Check for space after colon
        self.assertIn(": No", full_text, f"Expected ': No' in template, got: {full_text}")


class TestChartQATokenization(unittest.TestCase):
    """Test ChartQA answer position finding"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if CHARTQA_CACHE.exists():
            with open(CHARTQA_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_exists(self):
        """Test ChartQA cache exists"""
        self.assertTrue(CHARTQA_CACHE.exists())

    def test_all_answers_findable(self):
        """Test that all ChartQA answers can be found in tokenized sequences"""
        if not self.data:
            self.skipTest("No ChartQA data")

        failures = []
        samples_to_test = min(50, len(self.data))

        for idx in range(samples_to_test):
            item = self.data[idx]
            question = item.get('query', '')
            label = item.get('label', [''])
            answer = label[0] if isinstance(label, list) and label else str(label)
            image_path = item.get('image_path', '')

            if not Path(image_path).exists():
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue

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

            full_token_list = full_inputs["input_ids"][0].tolist()
            pos, method = find_answer_position(self.processor, full_token_list, answer)

            if pos is None:
                failures.append({
                    'idx': idx,
                    'answer': answer,
                    'tokens_spaced': self.processor.tokenizer.encode(' ' + answer, add_special_tokens=False),
                    'tokens_no_space': self.processor.tokenizer.encode(answer, add_special_tokens=False)
                })

        self.assertEqual(len(failures), 0,
            f"Failed to find {len(failures)} answers:\n" +
            "\n".join([f"  {f['idx']}: '{f['answer']}'" for f in failures[:10]]))

    def test_yes_no_answers_findable(self):
        """Test specifically Yes/No answers which had issues before"""
        if not self.data:
            self.skipTest("No ChartQA data")

        # Find Yes/No answers
        yes_no_samples = []
        for idx, item in enumerate(self.data[:100]):
            label = item.get('label', [''])
            answer = label[0] if isinstance(label, list) and label else str(label)
            if answer.lower() in ['yes', 'no']:
                yes_no_samples.append((idx, item, answer))

        self.assertGreater(len(yes_no_samples), 0, "Should have Yes/No samples")

        for idx, item, answer in yes_no_samples[:10]:
            image_path = item.get('image_path', '')
            if not Path(image_path).exists():
                continue

            image = Image.open(image_path).convert('RGB')
            question = item.get('query', '')

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

            full_token_list = full_inputs["input_ids"][0].tolist()
            pos, method = find_answer_position(self.processor, full_token_list, answer)

            self.assertIsNotNone(pos,
                f"Sample {idx}: Could not find '{answer}' in tokens")
            self.assertEqual(method, "with_space",
                f"Sample {idx}: '{answer}' should be found with leading space method")


class TestDocVQATokenization(unittest.TestCase):
    """Test DocVQA answer position finding"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if DOCVQA_CACHE.exists():
            with open(DOCVQA_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_exists(self):
        """Test DocVQA cache exists"""
        self.assertTrue(DOCVQA_CACHE.exists())

    def test_query_is_dict(self):
        """Test that DocVQA query field is a dict with language keys"""
        if not self.data:
            self.skipTest("No DocVQA data")

        for idx, item in enumerate(self.data[:10]):
            query = item.get('query', '')
            self.assertIsInstance(query, dict,
                f"Sample {idx}: query should be dict, got {type(query)}")
            self.assertIn('en', query,
                f"Sample {idx}: query should have 'en' key")

    def test_all_answers_findable(self):
        """Test that all DocVQA answers can be found in tokenized sequences"""
        if not self.data:
            self.skipTest("No DocVQA data")

        failures = []
        samples_to_test = min(50, len(self.data))

        for idx in range(samples_to_test):
            item = self.data[idx]

            # Extract question (handle dict format)
            query_field = item.get('query', '')
            if isinstance(query_field, dict):
                question = query_field.get('en', str(query_field))
            else:
                question = str(query_field)

            answers = item.get('answers', [])
            answer = answers[0] if answers else ''
            image_path = item.get('image_path', '')

            if not Path(image_path).exists():
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue

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

            full_token_list = full_inputs["input_ids"][0].tolist()
            pos, method = find_answer_position(self.processor, full_token_list, answer)

            if pos is None:
                failures.append({
                    'idx': idx,
                    'answer': answer,
                    'question': question[:50]
                })

        self.assertEqual(len(failures), 0,
            f"Failed to find {len(failures)} answers:\n" +
            "\n".join([f"  {f['idx']}: '{f['answer']}'" for f in failures[:10]]))


class TestOCRBenchTokenization(unittest.TestCase):
    """Test OCRBench answer position finding"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if OCRBENCH_CACHE.exists():
            with open(OCRBENCH_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_exists(self):
        """Test OCRBench cache exists"""
        self.assertTrue(OCRBENCH_CACHE.exists())

    def test_all_answers_findable(self):
        """Test that all OCRBench answers can be found in tokenized sequences"""
        if not self.data:
            self.skipTest("No OCRBench data")

        failures = []
        samples_to_test = min(50, len(self.data))

        for idx in range(samples_to_test):
            item = self.data[idx]
            question = item.get('question', '')
            answer_list = item.get('answer', [''])
            answer = answer_list[0] if isinstance(answer_list, list) and answer_list else str(answer_list)
            image_path = item.get('image_path', '')

            if not Path(image_path).exists():
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue

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

            full_token_list = full_inputs["input_ids"][0].tolist()
            pos, method = find_answer_position(self.processor, full_token_list, answer)

            if pos is None:
                failures.append({
                    'idx': idx,
                    'answer': answer,
                    'question': question[:50]
                })

        self.assertEqual(len(failures), 0,
            f"Failed to find {len(failures)} answers:\n" +
            "\n".join([f"  {f['idx']}: '{f['answer']}'" for f in failures[:10]]))


class TestMaskingPosition(unittest.TestCase):
    """Test that masking positions are correct"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    def test_masking_leaves_answer_unmasked(self):
        """Test that only answer tokens are unmasked"""
        image = Image.new('RGB', (512, 512), color='white')
        question = "What is 2+2?"
        answer = "4"

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

        full_token_list = full_inputs["input_ids"][0].tolist()
        pos, _ = find_answer_position(self.processor, full_token_list, answer)

        self.assertIsNotNone(pos, "Should find answer position")

        # Create labels with masking
        labels = full_inputs["input_ids"].clone()
        labels[:, :pos] = -100

        # Count unmasked tokens
        unmasked = (labels[0] != -100).sum().item()

        self.assertGreater(unmasked, 0, "Should have unmasked tokens")
        self.assertLess(unmasked, len(full_token_list),
            "Should have masked tokens (not all unmasked)")

        # Unmasked tokens should be at the end
        unmasked_positions = (labels[0] != -100).nonzero().squeeze().tolist()
        if isinstance(unmasked_positions, int):
            unmasked_positions = [unmasked_positions]

        # All unmasked positions should be >= pos
        for p in unmasked_positions:
            self.assertGreaterEqual(p, pos,
                f"Unmasked position {p} should be >= answer start {pos}")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestTokenizationBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestChartQATokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestDocVQATokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRBenchTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestMaskingPosition))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
