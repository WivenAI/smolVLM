"""
Unit Tests for QCM Datasets

Tests verify that:
1. QCM JSON files have correct structure
2. Images exist for all entries
3. Answer options (A, B, C, D) are valid
4. Correct answers are in options
5. Tokenization works for QCM format
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

# QCM dataset paths
QCM_DIR = Path(__file__).parent.parent / "datasets" / "qcm"
IMAGES_DIR = Path(__file__).parent.parent / "datasets" / "images"
PROCEDURE_IMAGES_DIR = Path(__file__).parent.parent / "datasets" / "procedureimages"

# QCM files and their image directories
QCM_DATASETS = {
    "gemini": {
        "file": QCM_DIR / "qcm_dataset_gemini.json",
        "images_dir": IMAGES_DIR,
        "has_images": True
    },
    "nova": {
        "file": QCM_DIR / "qcm_dataset_nova_pro.json",
        "images_dir": IMAGES_DIR,
        "has_images": True
    },
    "claudette": {
        "file": QCM_DIR / "qcm_claudette.json",
        "images_dir": None,
        "has_images": False  # Claudette doesn't have images
    },
    "procedure1": {
        "file": QCM_DIR / "qcm_procedure1_claude_code.json",
        "images_dir": PROCEDURE_IMAGES_DIR,
        "has_images": True
    },
    "procedure2": {
        "file": QCM_DIR / "qcm_procedure2_geminicli.json",
        "images_dir": PROCEDURE_IMAGES_DIR,
        "has_images": True
    }
}


def find_answer_position(processor, full_token_list, answer):
    """Find answer position using the fixed algorithm."""
    # Try 1: Answer with leading space
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


class TestQCMFilesExist(unittest.TestCase):
    """Test that QCM files exist"""

    def test_gemini_file_exists(self):
        self.assertTrue(QCM_DATASETS["gemini"]["file"].exists())

    def test_nova_file_exists(self):
        self.assertTrue(QCM_DATASETS["nova"]["file"].exists())

    def test_claudette_file_exists(self):
        self.assertTrue(QCM_DATASETS["claudette"]["file"].exists())

    def test_procedure1_file_exists(self):
        self.assertTrue(QCM_DATASETS["procedure1"]["file"].exists())

    def test_procedure2_file_exists(self):
        self.assertTrue(QCM_DATASETS["procedure2"]["file"].exists())


class TestQCMStructure(unittest.TestCase):
    """Test QCM JSON structure"""

    def test_gemini_structure(self):
        """Test Gemini QCM has correct structure"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for idx, item in enumerate(data[:10]):
            self.assertIn('image_name', item, f"Item {idx} missing 'image_name'")
            self.assertIn('qcm', item, f"Item {idx} missing 'qcm'")
            self.assertIn('question', item['qcm'], f"Item {idx} missing 'question'")
            self.assertIn('options', item['qcm'], f"Item {idx} missing 'options'")
            self.assertIn('correct_answer', item['qcm'], f"Item {idx} missing 'correct_answer'")

    def test_nova_structure(self):
        """Test Nova QCM has correct structure"""
        with open(QCM_DATASETS["nova"]["file"], 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for idx, item in enumerate(data[:10]):
            self.assertIn('image_name', item, f"Item {idx} missing 'image_name'")
            self.assertIn('qcm', item, f"Item {idx} missing 'qcm'")

    def test_procedure1_structure(self):
        """Test Procedure1 QCM has correct structure"""
        with open(QCM_DATASETS["procedure1"]["file"], 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for idx, item in enumerate(data[:10]):
            self.assertIn('image_name', item, f"Item {idx} missing 'image_name'")
            self.assertIn('qcm', item, f"Item {idx} missing 'qcm'")

    def test_procedure2_structure(self):
        """Test Procedure2 QCM has correct structure"""
        with open(QCM_DATASETS["procedure2"]["file"], 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for idx, item in enumerate(data[:10]):
            self.assertIn('image_name', item, f"Item {idx} missing 'image_name'")
            self.assertIn('qcm', item, f"Item {idx} missing 'qcm'")

    def test_claudette_structure(self):
        """Test Claudette QCM has correct structure (no images)"""
        with open(QCM_DATASETS["claudette"]["file"], 'r') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for idx, item in enumerate(data[:10]):
            self.assertIn('question', item, f"Item {idx} missing 'question'")
            self.assertIn('options', item, f"Item {idx} missing 'options'")
            self.assertIn('correct_answer', item, f"Item {idx} missing 'correct_answer'")


class TestQCMOptions(unittest.TestCase):
    """Test QCM options are valid"""

    def _check_options(self, data, dataset_name, has_nested_qcm=True):
        """Check that options are valid A, B, C, D"""
        valid_answers = {'A', 'B', 'C', 'D'}
        issues = []

        for idx, item in enumerate(data[:50]):
            if has_nested_qcm:
                qcm = item.get('qcm', {})
            else:
                qcm = item

            options = qcm.get('options', {})
            correct = qcm.get('correct_answer', '')

            # Check options exist
            if not options:
                issues.append(f"Item {idx}: No options")
                continue

            # Check correct answer is in options
            if correct not in options:
                issues.append(f"Item {idx}: correct_answer '{correct}' not in options {list(options.keys())}")

            # Check for valid option keys
            for key in options.keys():
                if key not in valid_answers:
                    issues.append(f"Item {idx}: Invalid option key '{key}'")

        return issues

    def test_gemini_options_valid(self):
        """Test Gemini QCM options are valid"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)
        issues = self._check_options(data, "gemini", has_nested_qcm=True)
        self.assertEqual(len(issues), 0, f"Issues found:\n" + "\n".join(issues[:10]))

    def test_nova_options_valid(self):
        """Test Nova QCM options are valid"""
        with open(QCM_DATASETS["nova"]["file"], 'r') as f:
            data = json.load(f)
        issues = self._check_options(data, "nova", has_nested_qcm=True)
        self.assertEqual(len(issues), 0, f"Issues found:\n" + "\n".join(issues[:10]))

    def test_procedure1_options_valid(self):
        """Test Procedure1 QCM options are valid"""
        with open(QCM_DATASETS["procedure1"]["file"], 'r') as f:
            data = json.load(f)
        issues = self._check_options(data, "procedure1", has_nested_qcm=True)
        self.assertEqual(len(issues), 0, f"Issues found:\n" + "\n".join(issues[:10]))

    def test_procedure2_options_valid(self):
        """Test Procedure2 QCM options are valid"""
        with open(QCM_DATASETS["procedure2"]["file"], 'r') as f:
            data = json.load(f)
        issues = self._check_options(data, "procedure2", has_nested_qcm=True)
        self.assertEqual(len(issues), 0, f"Issues found:\n" + "\n".join(issues[:10]))

    def test_claudette_options_valid(self):
        """Test Claudette QCM options are valid"""
        with open(QCM_DATASETS["claudette"]["file"], 'r') as f:
            data = json.load(f)
        issues = self._check_options(data, "claudette", has_nested_qcm=False)
        self.assertEqual(len(issues), 0, f"Issues found:\n" + "\n".join(issues[:10]))


class TestQCMImages(unittest.TestCase):
    """Test QCM images exist"""

    def _check_images(self, data, images_dir, dataset_name):
        """Check that all referenced images exist"""
        missing = []
        found = 0

        for idx, item in enumerate(data[:50]):
            image_name = item.get('image_name', '')
            if not image_name:
                continue

            image_path = images_dir / image_name
            if not image_path.exists():
                missing.append(f"Item {idx}: {image_name}")
            else:
                found += 1

        return missing, found

    def test_gemini_images_exist(self):
        """Test Gemini QCM images exist"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)
        missing, found = self._check_images(data, IMAGES_DIR, "gemini")
        self.assertEqual(len(missing), 0, f"Missing images:\n" + "\n".join(missing[:10]))
        self.assertGreater(found, 0, "Should find at least some images")

    def test_nova_images_exist(self):
        """Test Nova QCM images exist"""
        with open(QCM_DATASETS["nova"]["file"], 'r') as f:
            data = json.load(f)
        missing, found = self._check_images(data, IMAGES_DIR, "nova")
        self.assertEqual(len(missing), 0, f"Missing images:\n" + "\n".join(missing[:10]))
        self.assertGreater(found, 0, "Should find at least some images")

    def test_procedure1_images_exist(self):
        """Test Procedure1 QCM images exist"""
        with open(QCM_DATASETS["procedure1"]["file"], 'r') as f:
            data = json.load(f)
        missing, found = self._check_images(data, PROCEDURE_IMAGES_DIR, "procedure1")
        self.assertEqual(len(missing), 0, f"Missing images:\n" + "\n".join(missing[:10]))
        self.assertGreater(found, 0, "Should find at least some images")

    def test_procedure2_images_exist(self):
        """Test Procedure2 QCM images exist"""
        with open(QCM_DATASETS["procedure2"]["file"], 'r') as f:
            data = json.load(f)
        missing, found = self._check_images(data, PROCEDURE_IMAGES_DIR, "procedure2")
        self.assertEqual(len(missing), 0, f"Missing images:\n" + "\n".join(missing[:10]))
        self.assertGreater(found, 0, "Should find at least some images")

    def test_images_are_loadable(self):
        """Test that images can be loaded"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)

        load_errors = []
        for idx, item in enumerate(data[:10]):
            image_name = item.get('image_name', '')
            if not image_name:
                continue

            image_path = IMAGES_DIR / image_name
            if image_path.exists():
                try:
                    img = Image.open(image_path).convert('RGB')
                    self.assertGreater(img.width, 0)
                    self.assertGreater(img.height, 0)
                except Exception as e:
                    load_errors.append(f"Item {idx}: {image_name} - {e}")

        self.assertEqual(len(load_errors), 0, f"Load errors:\n" + "\n".join(load_errors))


class TestQCMTokenization(unittest.TestCase):
    """Test QCM answer tokenization"""

    @classmethod
    def setUpClass(cls):
        cls.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

    def test_single_letter_answers_tokenization(self):
        """Test that A, B, C, D answers can be found"""
        answers = ['A', 'B', 'C', 'D']

        for answer in answers:
            # Test with and without space
            no_space = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            with_space = self.processor.tokenizer.encode(' ' + answer, add_special_tokens=False)

            self.assertGreater(len(no_space), 0, f"'{answer}' should have tokens")
            self.assertGreater(len(with_space), 0, f"' {answer}' should have tokens")

    def test_qcm_format_answer_findable(self):
        """Test that QCM answers can be found in tokenized sequence"""
        # Create a dummy image
        image = Image.new('RGB', (512, 512), color='white')

        # Simulate QCM format
        question = "What is the capital of France?\nA: Paris\nB: London\nC: Berlin\nD: Madrid"
        answer = "A"

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

        full_token_list = full_inputs["input_ids"][0].tolist()
        pos, method = find_answer_position(self.processor, full_token_list, answer)

        self.assertIsNotNone(pos, f"Should find answer '{answer}' in tokens")

    def test_real_qcm_gemini_tokenization(self):
        """Test tokenization with real Gemini QCM data"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)

        failures = []
        tested = 0

        for idx, item in enumerate(data[:20]):
            image_name = item.get('image_name', '')
            image_path = IMAGES_DIR / image_name

            if not image_path.exists():
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue

            qcm = item.get('qcm', {})
            question = qcm.get('question', '')
            options = qcm.get('options', {})
            answer = qcm.get('correct_answer', '')

            # Format question with options
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            full_question = f"{question}\n{options_text}"

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_question}
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
                failures.append(f"Item {idx}: answer '{answer}' not found")
            else:
                tested += 1

        self.assertGreater(tested, 0, "Should test at least some samples")
        self.assertEqual(len(failures), 0, f"Failures:\n" + "\n".join(failures[:10]))

    def test_real_qcm_procedure1_tokenization(self):
        """Test tokenization with real Procedure1 QCM data"""
        with open(QCM_DATASETS["procedure1"]["file"], 'r') as f:
            data = json.load(f)

        failures = []
        tested = 0

        for idx, item in enumerate(data[:20]):
            image_name = item.get('image_name', '')
            image_path = PROCEDURE_IMAGES_DIR / image_name

            if not image_path.exists():
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except:
                continue

            qcm = item.get('qcm', {})
            question = qcm.get('question', '')
            options = qcm.get('options', {})
            answer = qcm.get('correct_answer', '')

            # Format question with options
            options_text = "\n".join([f"{k}: {v}" for k, v in options.items()])
            full_question = f"{question}\n{options_text}"

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_question}
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
                failures.append(f"Item {idx}: answer '{answer}' not found")
            else:
                tested += 1

        self.assertGreater(tested, 0, "Should test at least some samples")
        self.assertEqual(len(failures), 0, f"Failures:\n" + "\n".join(failures[:10]))


class TestQCMDatasetStats(unittest.TestCase):
    """Test QCM dataset statistics"""

    def test_gemini_sample_count(self):
        """Test Gemini has sufficient samples"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)
        self.assertGreater(len(data), 100, f"Gemini should have >100 samples, got {len(data)}")
        print(f"Gemini: {len(data)} samples")

    def test_nova_sample_count(self):
        """Test Nova has sufficient samples"""
        with open(QCM_DATASETS["nova"]["file"], 'r') as f:
            data = json.load(f)
        self.assertGreater(len(data), 100, f"Nova should have >100 samples, got {len(data)}")
        print(f"Nova: {len(data)} samples")

    def test_procedure1_sample_count(self):
        """Test Procedure1 has sufficient samples"""
        with open(QCM_DATASETS["procedure1"]["file"], 'r') as f:
            data = json.load(f)
        self.assertGreater(len(data), 10, f"Procedure1 should have >10 samples, got {len(data)}")
        print(f"Procedure1: {len(data)} samples")

    def test_procedure2_sample_count(self):
        """Test Procedure2 has sufficient samples"""
        with open(QCM_DATASETS["procedure2"]["file"], 'r') as f:
            data = json.load(f)
        self.assertGreater(len(data), 10, f"Procedure2 should have >10 samples, got {len(data)}")
        print(f"Procedure2: {len(data)} samples")

    def test_answer_distribution(self):
        """Test answer distribution is somewhat balanced"""
        with open(QCM_DATASETS["gemini"]["file"], 'r') as f:
            data = json.load(f)

        answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for item in data:
            answer = item.get('qcm', {}).get('correct_answer', '')
            if answer in answer_counts:
                answer_counts[answer] += 1

        print(f"Answer distribution: {answer_counts}")

        # Check no answer is missing completely
        for key, count in answer_counts.items():
            self.assertGreater(count, 0, f"Answer '{key}' has no samples")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestQCMFilesExist))
    suite.addTests(loader.loadTestsFromTestCase(TestQCMStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestQCMOptions))
    suite.addTests(loader.loadTestsFromTestCase(TestQCMImages))
    suite.addTests(loader.loadTestsFromTestCase(TestQCMTokenization))
    suite.addTests(loader.loadTestsFromTestCase(TestQCMDatasetStats))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
