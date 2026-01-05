"""
Unit Tests for Cached Dataset Functions

Tests verify that:
1. Dataset cache files have correct structure
2. Field names match what evaluators expect
3. Image paths exist and are loadable
4. Answer formats are correctly handled (lists vs strings)
5. Multi-language query handling works properly
"""

import unittest
import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset cache paths
CACHE_DIR = Path(__file__).parent.parent / "datasets" / "cache"
CHARTQA_CACHE = CACHE_DIR / "HuggingFaceM4_ChartQA_test.json"
OCRBENCH_CACHE = CACHE_DIR / "echo840_OCRBench_test.json"
DOCVQA_CACHE = CACHE_DIR / "nielsr_docvqa_1200_examples_train.json"


class TestChartQADataset(unittest.TestCase):
    """Test ChartQA cached dataset structure"""

    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests"""
        if CHARTQA_CACHE.exists():
            with open(CHARTQA_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_file_exists(self):
        """Test that cache file exists"""
        self.assertTrue(CHARTQA_CACHE.exists(), f"ChartQA cache not found at {CHARTQA_CACHE}")

    def test_data_not_empty(self):
        """Test that dataset has samples"""
        self.assertGreater(len(self.data), 0, "ChartQA dataset should not be empty")

    def test_required_fields_present(self):
        """Test that all required fields are present"""
        if not self.data:
            self.skipTest("No data loaded")

        required_fields = ['query', 'label', 'image_path']
        for idx, item in enumerate(self.data[:10]):
            for field in required_fields:
                self.assertIn(field, item, f"Sample {idx} missing field '{field}'")

    def test_query_field_is_string(self):
        """Test that query field is a string (not dict like DocVQA)"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIsInstance(item['query'], str,
                f"Sample {idx}: query should be string, got {type(item['query'])}")

    def test_label_field_is_list(self):
        """Test that label field is a list"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIsInstance(item['label'], list,
                f"Sample {idx}: label should be list, got {type(item['label'])}")
            self.assertGreater(len(item['label']), 0,
                f"Sample {idx}: label list should not be empty")

    def test_image_paths_exist(self):
        """Test that image files exist on disk"""
        if not self.data:
            self.skipTest("No data loaded")

        missing_images = []
        for idx, item in enumerate(self.data[:20]):
            img_path = Path(item['image_path'])
            if not img_path.exists():
                missing_images.append((idx, str(img_path)))

        self.assertEqual(len(missing_images), 0,
            f"Missing images: {missing_images[:5]}...")

    def test_images_are_loadable(self):
        """Test that images can be loaded as PIL Images"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:5]):
            img_path = item['image_path']
            try:
                img = Image.open(img_path).convert('RGB')
                self.assertIsNotNone(img)
                self.assertGreater(img.width, 0)
                self.assertGreater(img.height, 0)
            except Exception as e:
                self.fail(f"Failed to load image {idx} at {img_path}: {e}")

    def test_evaluator_field_access_pattern(self):
        """Test that evaluator's field access pattern works correctly"""
        if not self.data:
            self.skipTest("No data loaded")

        # From evaluator_chartqa.py:
        # question = item.get('question', item.get('query', ''))
        # answer = item.get('answer', item.get('label', ''))

        for idx, item in enumerate(self.data[:10]):
            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('label', ''))

            self.assertIsInstance(question, str, f"Sample {idx}: question should be string")
            self.assertTrue(len(question) > 0, f"Sample {idx}: question should not be empty")

            # Note: answer could be a list (label is a list)
            # The evaluator should handle this properly
            self.assertIsNotNone(answer, f"Sample {idx}: answer should not be None")


class TestOCRBenchDataset(unittest.TestCase):
    """Test OCRBench cached dataset structure"""

    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests"""
        if OCRBENCH_CACHE.exists():
            with open(OCRBENCH_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_file_exists(self):
        """Test that cache file exists"""
        self.assertTrue(OCRBENCH_CACHE.exists(), f"OCRBench cache not found at {OCRBENCH_CACHE}")

    def test_data_not_empty(self):
        """Test that dataset has samples"""
        self.assertGreater(len(self.data), 0, "OCRBench dataset should not be empty")

    def test_required_fields_present(self):
        """Test that all required fields are present"""
        if not self.data:
            self.skipTest("No data loaded")

        required_fields = ['question', 'answer', 'image_path']
        for idx, item in enumerate(self.data[:10]):
            for field in required_fields:
                self.assertIn(field, item, f"Sample {idx} missing field '{field}'")

    def test_question_field_is_string(self):
        """Test that question field is a string"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIsInstance(item['question'], str,
                f"Sample {idx}: question should be string, got {type(item['question'])}")

    def test_answer_field_is_list(self):
        """Test that answer field is a list"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIsInstance(item['answer'], list,
                f"Sample {idx}: answer should be list, got {type(item['answer'])}")

    def test_question_type_present(self):
        """Test that question_type field is present"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIn('question_type', item, f"Sample {idx} missing 'question_type'")

    def test_image_paths_exist(self):
        """Test that image files exist on disk"""
        if not self.data:
            self.skipTest("No data loaded")

        missing_images = []
        for idx, item in enumerate(self.data[:20]):
            img_path = Path(item['image_path'])
            if not img_path.exists():
                missing_images.append((idx, str(img_path)))

        self.assertEqual(len(missing_images), 0,
            f"Missing images: {missing_images[:5]}...")

    def test_images_are_loadable(self):
        """Test that images can be loaded as PIL Images"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:5]):
            img_path = item['image_path']
            try:
                img = Image.open(img_path).convert('RGB')
                self.assertIsNotNone(img)
            except Exception as e:
                self.fail(f"Failed to load image {idx} at {img_path}: {e}")

    def test_evaluator_field_access_pattern(self):
        """Test that evaluator's field access pattern works correctly"""
        if not self.data:
            self.skipTest("No data loaded")

        # From evaluator_ocr.py:
        # question = item['question']
        # ground_truth = item.get('answer', '')

        for idx, item in enumerate(self.data[:10]):
            question = item['question']
            ground_truth = item.get('answer', '')

            self.assertIsInstance(question, str, f"Sample {idx}: question should be string")
            self.assertTrue(len(question) > 0, f"Sample {idx}: question should not be empty")
            self.assertIsNotNone(ground_truth, f"Sample {idx}: answer should not be None")


class TestDocVQADataset(unittest.TestCase):
    """Test DocVQA cached dataset structure"""

    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests"""
        if DOCVQA_CACHE.exists():
            with open(DOCVQA_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_cache_file_exists(self):
        """Test that cache file exists"""
        self.assertTrue(DOCVQA_CACHE.exists(), f"DocVQA cache not found at {DOCVQA_CACHE}")

    def test_data_not_empty(self):
        """Test that dataset has samples"""
        self.assertGreater(len(self.data), 0, "DocVQA dataset should not be empty")

    def test_required_fields_present(self):
        """Test that all required fields are present"""
        if not self.data:
            self.skipTest("No data loaded")

        required_fields = ['query', 'answers', 'image_path']
        for idx, item in enumerate(self.data[:10]):
            for field in required_fields:
                self.assertIn(field, item, f"Sample {idx} missing field '{field}'")

    def test_query_field_is_dict_with_languages(self):
        """Test that query field is a dict with language keys"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            query = item['query']
            self.assertIsInstance(query, dict,
                f"Sample {idx}: query should be dict, got {type(query)}")
            self.assertIn('en', query,
                f"Sample {idx}: query dict should have 'en' key")

    def test_query_english_text_is_string(self):
        """Test that English query text is a non-empty string"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            en_query = item['query'].get('en', '')
            self.assertIsInstance(en_query, str,
                f"Sample {idx}: English query should be string")
            self.assertTrue(len(en_query) > 0,
                f"Sample {idx}: English query should not be empty")

    def test_answers_field_is_list(self):
        """Test that answers field is a list"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            self.assertIsInstance(item['answers'], list,
                f"Sample {idx}: answers should be list, got {type(item['answers'])}")

    def test_image_paths_exist(self):
        """Test that image files exist on disk"""
        if not self.data:
            self.skipTest("No data loaded")

        missing_images = []
        for idx, item in enumerate(self.data[:20]):
            img_path = Path(item['image_path'])
            if not img_path.exists():
                missing_images.append((idx, str(img_path)))

        self.assertEqual(len(missing_images), 0,
            f"Missing images: {missing_images[:5]}...")

    def test_images_are_loadable(self):
        """Test that images can be loaded as PIL Images"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:5]):
            img_path = item['image_path']
            try:
                img = Image.open(img_path).convert('RGB')
                self.assertIsNotNone(img)
            except Exception as e:
                self.fail(f"Failed to load image {idx} at {img_path}: {e}")

    def test_evaluator_field_access_pattern_bug(self):
        """
        Test that evaluator's field access pattern works correctly

        POTENTIAL BUG: The evaluator uses item.get('question', item.get('query', ''))
        but 'query' is a dict, not a string! This test documents this issue.
        """
        if not self.data:
            self.skipTest("No data loaded")

        # From evaluator_docvqa.py:
        # question = item.get('question', item.get('query', ''))
        # This will get the dict, not the English text!

        for idx, item in enumerate(self.data[:10]):
            question = item.get('question', item.get('query', ''))

            # This is the BUG: question will be a dict, not a string
            if isinstance(question, dict):
                # The correct way to get the question
                correct_question = question.get('en', '')
                self.assertIsInstance(correct_question, str)
                self.assertTrue(len(correct_question) > 0)

                # Document the bug
                print(f"\nWARNING: Sample {idx} - evaluator gets dict instead of string:")
                print(f"  item.get('question', item.get('query', '')) returns: {type(question)}")
                print(f"  Correct approach: item['query'].get('en', '') = '{correct_question[:50]}...'")


class TestChartQAEvaluatorCompatibility(unittest.TestCase):
    """Test that ChartQA evaluator handles dataset format correctly"""

    @classmethod
    def setUpClass(cls):
        """Load dataset once for all tests"""
        if CHARTQA_CACHE.exists():
            with open(CHARTQA_CACHE, 'r') as f:
                cls.data = json.load(f)
        else:
            cls.data = []

    def test_label_list_to_string_conversion(self):
        """Test that label list is properly converted for comparison"""
        if not self.data:
            self.skipTest("No data loaded")

        for idx, item in enumerate(self.data[:10]):
            label = item.get('label', [])

            # The evaluator gets: answer = item.get('answer', item.get('label', ''))
            # This returns a list, not a string
            answer = item.get('answer', item.get('label', ''))

            if isinstance(answer, list):
                # Need to extract first element or join
                answer_str = str(answer[0]) if answer else ''
                self.assertIsInstance(answer_str, str)
                print(f"Sample {idx}: label is list {label}, should use {answer_str}")


class TestAnswerNormalization(unittest.TestCase):
    """Test answer normalization for accuracy calculation"""

    def test_normalize_text_import(self):
        """Test that normalize_text can be imported"""
        try:
            from evaluators.qcm_accuracy import normalize_text
            self.assertTrue(callable(normalize_text))
        except ImportError as e:
            self.fail(f"Could not import normalize_text: {e}")

    def test_normalize_text_basic(self):
        """Test basic text normalization"""
        from evaluators.qcm_accuracy import normalize_text

        # Test basic normalization
        self.assertEqual(normalize_text("Hello World"), "helloworld")
        self.assertEqual(normalize_text("  UPPER  "), "upper")
        self.assertEqual(normalize_text("No."), "no")

    def test_normalize_text_numbers(self):
        """Test number normalization"""
        from evaluators.qcm_accuracy import normalize_text

        self.assertEqual(normalize_text("42"), "42")
        self.assertEqual(normalize_text("0.57"), "057")  # or depends on implementation
        self.assertEqual(normalize_text("$100"), "100")

    def test_list_answer_handling(self):
        """Test handling of list answers"""
        # Simulate what evaluators should do
        answers = ["14"]

        # Convert list to comparable format
        if isinstance(answers, list) and len(answers) > 0:
            answer = str(answers[0])
        else:
            answer = str(answers)

        self.assertEqual(answer, "14")


class TestDatasetStatistics(unittest.TestCase):
    """Test dataset statistics and coverage"""

    def test_chartqa_sample_count(self):
        """Test ChartQA has sufficient samples"""
        if CHARTQA_CACHE.exists():
            with open(CHARTQA_CACHE, 'r') as f:
                data = json.load(f)
            self.assertGreater(len(data), 100, "ChartQA should have >100 samples")
            print(f"ChartQA: {len(data)} samples")

    def test_ocrbench_sample_count(self):
        """Test OCRBench has sufficient samples"""
        if OCRBENCH_CACHE.exists():
            with open(OCRBENCH_CACHE, 'r') as f:
                data = json.load(f)
            self.assertGreater(len(data), 100, "OCRBench should have >100 samples")
            print(f"OCRBench: {len(data)} samples")

    def test_docvqa_sample_count(self):
        """Test DocVQA has sufficient samples"""
        if DOCVQA_CACHE.exists():
            with open(DOCVQA_CACHE, 'r') as f:
                data = json.load(f)
            self.assertGreater(len(data), 100, "DocVQA should have >100 samples")
            print(f"DocVQA: {len(data)} samples")

    def test_ocrbench_question_types(self):
        """Test OCRBench has diverse question types"""
        if OCRBENCH_CACHE.exists():
            with open(OCRBENCH_CACHE, 'r') as f:
                data = json.load(f)

            question_types = set()
            for item in data:
                qt = item.get('question_type', 'unknown')
                question_types.add(qt)

            print(f"OCRBench question types: {question_types}")
            self.assertGreater(len(question_types), 1, "Should have multiple question types")


class TestImagePathConsistency(unittest.TestCase):
    """Test that image paths are consistent across datasets"""

    def test_all_chartqa_images_in_cache_dir(self):
        """Test all ChartQA images are in cache directory"""
        if CHARTQA_CACHE.exists():
            with open(CHARTQA_CACHE, 'r') as f:
                data = json.load(f)

            for idx, item in enumerate(data[:50]):
                img_path = Path(item['image_path'])
                self.assertTrue(
                    str(CACHE_DIR) in str(img_path),
                    f"Image {idx} path not in cache dir: {img_path}"
                )

    def test_all_ocrbench_images_in_cache_dir(self):
        """Test all OCRBench images are in cache directory"""
        if OCRBENCH_CACHE.exists():
            with open(OCRBENCH_CACHE, 'r') as f:
                data = json.load(f)

            for idx, item in enumerate(data[:50]):
                img_path = Path(item['image_path'])
                self.assertTrue(
                    str(CACHE_DIR) in str(img_path),
                    f"Image {idx} path not in cache dir: {img_path}"
                )

    def test_all_docvqa_images_in_cache_dir(self):
        """Test all DocVQA images are in cache directory"""
        if DOCVQA_CACHE.exists():
            with open(DOCVQA_CACHE, 'r') as f:
                data = json.load(f)

            for idx, item in enumerate(data[:50]):
                img_path = Path(item['image_path'])
                self.assertTrue(
                    str(CACHE_DIR) in str(img_path),
                    f"Image {idx} path not in cache dir: {img_path}"
                )


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes in order of importance
    suite.addTests(loader.loadTestsFromTestCase(TestChartQADataset))
    suite.addTests(loader.loadTestsFromTestCase(TestOCRBenchDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestDocVQADataset))
    suite.addTests(loader.loadTestsFromTestCase(TestChartQAEvaluatorCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestAnswerNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestImagePathConsistency))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
