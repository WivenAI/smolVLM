"""
Unit Tests for DualLogger

These tests verify that:
1. DualLogger initializes correctly with various argument combinations
2. init_dual_logger passes arguments correctly (fixes the bool-as-path bug)
3. Path creation works correctly for tensorboard and wandb directories

Bug fixed: init_dual_logger was passing use_wandb (bool) as wandb_dir (path),
causing TypeError: argument should be a str or an os.PathLike object, not 'bool'
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dual_logger import DualLogger, init_dual_logger, get_dual_logger, log_metrics


class TestDualLoggerInit(unittest.TestCase):
    """Test DualLogger initialization"""

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_tensorboard_dir_only(self):
        """Test initialization with just tensorboard_dir"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir)

        self.assertEqual(str(logger.tensorboard_dir), tb_dir)
        self.assertTrue(logger.tensorboard_dir.exists())
        # wandb_dir should be sibling of tensorboard_dir
        expected_wandb = Path(tb_dir).parent / "wandb"
        self.assertEqual(logger.wandb_dir, expected_wandb)

    def test_init_with_use_wandb_false(self):
        """Test initialization with use_wandb=False"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=False)

        self.assertFalse(logger.use_wandb)
        self.assertTrue(logger.tensorboard_dir.exists())

    def test_init_with_use_wandb_true(self):
        """Test initialization with use_wandb=True (default)"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=True)

        # use_wandb might be True or False depending on wandb availability
        self.assertTrue(logger.tensorboard_dir.exists())

    def test_init_with_custom_wandb_dir(self):
        """Test initialization with custom wandb_dir"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        wandb_dir = os.path.join(self.temp_dir, "custom_wandb")

        logger = DualLogger(tb_dir, wandb_dir=wandb_dir, use_wandb=True)

        self.assertEqual(str(logger.wandb_dir), wandb_dir)
        self.assertTrue(logger.wandb_dir.exists())

    def test_init_with_all_arguments(self):
        """Test initialization with all arguments specified"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        wandb_dir = os.path.join(self.temp_dir, "wandb")

        logger = DualLogger(tb_dir, wandb_dir=wandb_dir, use_wandb=False)

        self.assertEqual(str(logger.tensorboard_dir), tb_dir)
        self.assertEqual(str(logger.wandb_dir), wandb_dir)
        self.assertFalse(logger.use_wandb)


class TestInitDualLoggerBugFix(unittest.TestCase):
    """Test that init_dual_logger passes arguments correctly

    This specifically tests the bug fix where use_wandb (bool) was being
    passed as wandb_dir (path), causing:
    TypeError: argument should be a str or an os.PathLike object, not 'bool'
    """

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_dual_logger_with_use_wandb_true(self):
        """Test init_dual_logger with use_wandb=True doesn't crash"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")

        # This should NOT raise TypeError
        logger = init_dual_logger(tb_dir, use_wandb=True)

        self.assertIsNotNone(logger)
        self.assertTrue(logger.tensorboard_dir.exists())
        # wandb_dir should be a path, not True
        self.assertIsInstance(logger.wandb_dir, Path)
        self.assertNotEqual(str(logger.wandb_dir), "True")

    def test_init_dual_logger_with_use_wandb_false(self):
        """Test init_dual_logger with use_wandb=False doesn't crash"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard2")

        # This should NOT raise TypeError
        logger = init_dual_logger(tb_dir, use_wandb=False)

        self.assertIsNotNone(logger)
        self.assertFalse(logger.use_wandb)
        # wandb_dir should be a path, not False
        self.assertIsInstance(logger.wandb_dir, Path)
        self.assertNotEqual(str(logger.wandb_dir), "False")

    def test_init_dual_logger_default_use_wandb(self):
        """Test init_dual_logger with default use_wandb"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard3")

        # This should NOT raise TypeError
        logger = init_dual_logger(tb_dir)

        self.assertIsNotNone(logger)
        self.assertIsInstance(logger.wandb_dir, Path)

    def test_wandb_dir_is_valid_path(self):
        """Test that wandb_dir is always a valid path, never a bool"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard4")

        for use_wandb in [True, False]:
            logger = init_dual_logger(tb_dir + str(use_wandb), use_wandb=use_wandb)

            # wandb_dir must be a Path
            self.assertIsInstance(logger.wandb_dir, Path)

            # wandb_dir string should not be "True" or "False"
            wandb_str = str(logger.wandb_dir)
            self.assertNotIn("True", wandb_str)
            self.assertNotIn("False", wandb_str)

            # wandb_dir should contain "wandb" in the path
            self.assertIn("wandb", wandb_str)


class TestBoolAsPathError(unittest.TestCase):
    """Test that passing bool as path raises appropriate error

    Documents the bug behavior to prevent regression.
    """

    def test_path_with_bool_raises_error(self):
        """Document that Path(bool) raises TypeError"""
        with self.assertRaises(TypeError) as ctx:
            Path(True)

        self.assertIn("bool", str(ctx.exception).lower())

    def test_dual_logger_with_bool_wandb_dir_raises_error(self):
        """Test that passing bool as wandb_dir raises TypeError"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tb_dir = os.path.join(temp_dir, "tensorboard")

            # This SHOULD raise TypeError - documenting the bug
            with self.assertRaises(TypeError) as ctx:
                # Directly calling DualLogger with bool as wandb_dir
                DualLogger(tb_dir, True, True)  # Second arg is wandb_dir

            self.assertIn("bool", str(ctx.exception).lower())


class TestGetDualLogger(unittest.TestCase):
    """Test get_dual_logger function"""

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_dual_logger_after_init(self):
        """Test that get_dual_logger returns the initialized logger"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")

        created_logger = init_dual_logger(tb_dir, use_wandb=False)
        retrieved_logger = get_dual_logger()

        self.assertIs(created_logger, retrieved_logger)


class TestLogMetrics(unittest.TestCase):
    """Test log_metrics function"""

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_metrics_with_initialized_logger(self):
        """Test logging metrics after initialization"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")

        init_dual_logger(tb_dir, use_wandb=False)

        # Should not raise any errors
        log_metrics({"loss": 0.5, "accuracy": 0.95}, step=1)
        log_metrics({"loss": 0.3}, step=2, commit=False)


class TestDualLoggerMethods(unittest.TestCase):
    """Test DualLogger methods"""

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_log_method(self):
        """Test the log method"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=False)

        # Should not raise any errors
        logger.log({"metric1": 1.0, "metric2": 2.0}, step=0)
        logger.log({"metric1": 0.5}, step=1, commit=True)

    def test_log_text_method(self):
        """Test the log_text method"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=False)

        # Should not raise any errors
        logger.log_text("test_tag", "test content", step=0)

    def test_flush_method(self):
        """Test the flush method"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=False)

        # Should not raise any errors
        logger.flush()

    def test_close_method(self):
        """Test the close method"""
        tb_dir = os.path.join(self.temp_dir, "tensorboard")
        logger = DualLogger(tb_dir, use_wandb=False)

        # Should not raise any errors
        logger.close()


class TestArgumentOrdering(unittest.TestCase):
    """Test correct argument ordering in DualLogger

    Ensures that positional and keyword arguments work correctly.
    """

    def setUp(self):
        """Create temp directory for tests"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_positional_args(self):
        """Test DualLogger with positional arguments"""
        tb_dir = os.path.join(self.temp_dir, "tb1")
        wandb_dir = os.path.join(self.temp_dir, "wandb1")

        # All positional: tensorboard_dir, wandb_dir, use_wandb
        logger = DualLogger(tb_dir, wandb_dir, False)

        self.assertEqual(str(logger.tensorboard_dir), tb_dir)
        self.assertEqual(str(logger.wandb_dir), wandb_dir)
        self.assertFalse(logger.use_wandb)

    def test_keyword_args(self):
        """Test DualLogger with keyword arguments"""
        tb_dir = os.path.join(self.temp_dir, "tb2")
        wandb_dir = os.path.join(self.temp_dir, "wandb2")

        # All keyword
        logger = DualLogger(
            tensorboard_dir=tb_dir,
            wandb_dir=wandb_dir,
            use_wandb=False
        )

        self.assertEqual(str(logger.tensorboard_dir), tb_dir)
        self.assertEqual(str(logger.wandb_dir), wandb_dir)
        self.assertFalse(logger.use_wandb)

    def test_mixed_args_correct(self):
        """Test DualLogger with mixed positional and keyword arguments (correct)"""
        tb_dir = os.path.join(self.temp_dir, "tb3")

        # tensorboard_dir positional, use_wandb keyword (skipping wandb_dir)
        logger = DualLogger(tb_dir, use_wandb=False)

        self.assertEqual(str(logger.tensorboard_dir), tb_dir)
        self.assertFalse(logger.use_wandb)
        # wandb_dir should default to parent/wandb
        self.assertIn("wandb", str(logger.wandb_dir))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDualLoggerInit))
    suite.addTests(loader.loadTestsFromTestCase(TestInitDualLoggerBugFix))
    suite.addTests(loader.loadTestsFromTestCase(TestBoolAsPathError))
    suite.addTests(loader.loadTestsFromTestCase(TestGetDualLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestLogMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDualLoggerMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestArgumentOrdering))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
