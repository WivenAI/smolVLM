"""
Unit Tests for Configuration Path Validation

These tests verify that:
1. Config path parameters are valid strings (not bool, None, or other types)
2. Path validation catches common config errors before training starts
3. The pipeline properly handles and reports path configuration issues

These tests address errors found in IZAR logs:
- "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'bool'"
- "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'"
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project root (relative to this test file)
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def validate_path_value(value: Any, param_name: str) -> tuple[bool, str]:
    """
    Validate that a path config value is a valid string or None.

    Returns:
        (is_valid, error_message)
    """
    if value is None:
        return True, ""  # None is acceptable (uses default)

    if isinstance(value, bool):
        return False, f"{param_name}: expected path string, got bool ({value})"

    if isinstance(value, (int, float)):
        return False, f"{param_name}: expected path string, got number ({value})"

    if isinstance(value, list):
        return False, f"{param_name}: expected path string, got list"

    if isinstance(value, dict):
        return False, f"{param_name}: expected path string, got dict"

    if not isinstance(value, str):
        return False, f"{param_name}: expected path string, got {type(value).__name__}"

    return True, ""


def validate_config_paths(config: Dict[str, Any]) -> list[str]:
    """
    Validate all path-related config values.

    Returns list of error messages (empty if all valid).
    """
    errors = []

    # Model config paths
    if "model" in config:
        model = config["model"]
        cache_dir = model.get("cache_dir")
        valid, err = validate_path_value(cache_dir, "model.cache_dir")
        if not valid:
            errors.append(err)

    # Paths config
    if "paths" in config:
        paths = config["paths"]
        for key in ["output_dir", "cache_dir", "results_dir", "logs_dir"]:
            if key in paths:
                valid, err = validate_path_value(paths[key], f"paths.{key}")
                if not valid:
                    errors.append(err)

    # Training strategies - check dataset and image_dir paths
    if "training" in config and "strategies" in config["training"]:
        for idx, strategy in enumerate(config["training"]["strategies"]):
            name = strategy.get("name", f"strategy_{idx}")

            if "dataset" in strategy:
                valid, err = validate_path_value(strategy["dataset"], f"strategies[{name}].dataset")
                if not valid:
                    errors.append(err)

            if "image_dir" in strategy:
                valid, err = validate_path_value(strategy["image_dir"], f"strategies[{name}].image_dir")
                if not valid:
                    errors.append(err)

    return errors


class TestPathValueValidation(unittest.TestCase):
    """Test the path value validation function"""

    def test_valid_string_path(self):
        """Test that string paths are valid"""
        valid, err = validate_path_value("datasets/cache", "test_param")
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_valid_absolute_path(self):
        """Test that absolute paths are valid"""
        valid, err = validate_path_value("/scratch/izar/cache", "test_param")
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_valid_relative_path(self):
        """Test that relative paths are valid"""
        valid, err = validate_path_value("../tmpcache", "test_param")
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_valid_none(self):
        """Test that None is valid (uses defaults)"""
        valid, err = validate_path_value(None, "test_param")
        self.assertTrue(valid)
        self.assertEqual(err, "")

    def test_invalid_bool_true(self):
        """Test that True is invalid"""
        valid, err = validate_path_value(True, "test_param")
        self.assertFalse(valid)
        self.assertIn("bool", err)
        self.assertIn("True", err)

    def test_invalid_bool_false(self):
        """Test that False is invalid"""
        valid, err = validate_path_value(False, "test_param")
        self.assertFalse(valid)
        self.assertIn("bool", err)
        self.assertIn("False", err)

    def test_invalid_integer(self):
        """Test that integers are invalid"""
        valid, err = validate_path_value(123, "test_param")
        self.assertFalse(valid)
        self.assertIn("number", err)

    def test_invalid_float(self):
        """Test that floats are invalid"""
        valid, err = validate_path_value(1.5, "test_param")
        self.assertFalse(valid)
        self.assertIn("number", err)

    def test_invalid_list(self):
        """Test that lists are invalid"""
        valid, err = validate_path_value(["path1", "path2"], "test_param")
        self.assertFalse(valid)
        self.assertIn("list", err)

    def test_invalid_dict(self):
        """Test that dicts are invalid"""
        valid, err = validate_path_value({"path": "value"}, "test_param")
        self.assertFalse(valid)
        self.assertIn("dict", err)


class TestConfigPathValidation(unittest.TestCase):
    """Test full config path validation"""

    def test_valid_config(self):
        """Test validation of a valid config"""
        config = {
            "model": {
                "base_model": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
                "cache_dir": "../tmpcache"
            },
            "paths": {
                "output_dir": "modelweights",
                "cache_dir": "datasets/cache",
                "results_dir": "results",
                "logs_dir": "logs"
            },
            "training": {
                "strategies": [
                    {"name": "baseline", "enabled": True, "type": "none"},
                    {"name": "sft_test", "dataset": "datasets/test.json", "image_dir": "datasets/images"}
                ]
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(errors, [])

    def test_bool_cache_dir(self):
        """Test detection of bool cache_dir (common YAML mistake)"""
        config = {
            "model": {
                "cache_dir": True  # YAML parses 'true' as boolean
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("model.cache_dir", errors[0])
        self.assertIn("bool", errors[0])

    def test_bool_output_dir(self):
        """Test detection of bool output_dir"""
        config = {
            "paths": {
                "output_dir": False
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("paths.output_dir", errors[0])

    def test_none_paths_valid(self):
        """Test that None/missing paths are valid (use defaults)"""
        config = {
            "paths": {
                "output_dir": None,
                "cache_dir": None
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(errors, [])

    def test_strategy_bool_dataset(self):
        """Test detection of bool in strategy dataset path"""
        config = {
            "training": {
                "strategies": [
                    {"name": "bad_strategy", "dataset": True}
                ]
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("bad_strategy", errors[0])
        self.assertIn("dataset", errors[0])

    def test_strategy_bool_image_dir(self):
        """Test detection of bool in strategy image_dir path"""
        config = {
            "training": {
                "strategies": [
                    {"name": "bad_strategy", "image_dir": False}
                ]
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("bad_strategy", errors[0])
        self.assertIn("image_dir", errors[0])

    def test_multiple_errors(self):
        """Test detection of multiple path errors"""
        config = {
            "model": {"cache_dir": True},
            "paths": {"output_dir": False, "results_dir": 123},
            "training": {
                "strategies": [
                    {"name": "strat1", "dataset": True},
                    {"name": "strat2", "image_dir": ["list", "of", "paths"]}
                ]
            }
        }
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 5)


class TestYAMLPathParsing(unittest.TestCase):
    """Test YAML parsing of path-like values"""

    def test_yaml_true_false_parsing(self):
        """Test that YAML parses 'true'/'false' as booleans, not strings"""
        yaml_content = """
model:
  cache_dir: true
  other_path: false
"""
        config = yaml.safe_load(yaml_content)

        # YAML parses these as booleans!
        self.assertIsInstance(config["model"]["cache_dir"], bool)
        self.assertTrue(config["model"]["cache_dir"])
        self.assertIsInstance(config["model"]["other_path"], bool)
        self.assertFalse(config["model"]["other_path"])

        # Validation should catch this
        errors = validate_config_paths(config)
        self.assertEqual(len(errors), 1)
        self.assertIn("bool", errors[0])

    def test_yaml_quoted_true_is_string(self):
        """Test that quoted 'true' in YAML remains a string"""
        yaml_content = """
model:
  cache_dir: "true"
"""
        config = yaml.safe_load(yaml_content)

        # Quoted values are strings
        self.assertIsInstance(config["model"]["cache_dir"], str)
        self.assertEqual(config["model"]["cache_dir"], "true")

        # This should be valid (weird but valid path)
        errors = validate_config_paths(config)
        self.assertEqual(errors, [])

    def test_yaml_null_parsing(self):
        """Test that YAML parses 'null' and '~' as None"""
        yaml_content = """
paths:
  output_dir: null
  cache_dir: ~
"""
        config = yaml.safe_load(yaml_content)

        self.assertIsNone(config["paths"]["output_dir"])
        self.assertIsNone(config["paths"]["cache_dir"])

        # None is acceptable (uses defaults)
        errors = validate_config_paths(config)
        self.assertEqual(errors, [])


class TestRealConfigFiles(unittest.TestCase):
    """Test actual config files in the project"""

    def test_main_config_paths(self):
        """Test that main conf.yaml has valid paths"""
        config_path = CONFIG_DIR / "conf.yaml"
        if not config_path.exists():
            self.skipTest(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        errors = validate_config_paths(config)
        self.assertEqual(errors, [], f"Path errors in conf.yaml: {errors}")

    def test_all_config_files(self):
        """Test all YAML config files in config directory"""
        if not CONFIG_DIR.exists():
            self.skipTest(f"Config directory not found: {CONFIG_DIR}")

        config_files = list(CONFIG_DIR.glob("*.yaml")) + list(CONFIG_DIR.glob("*.yml"))

        all_errors = {}
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    errors = validate_config_paths(config)
                    if errors:
                        all_errors[config_file.name] = errors
            except Exception as e:
                all_errors[config_file.name] = [f"Failed to parse: {e}"]

        if all_errors:
            error_msg = "\n".join(
                f"  {name}: {errs}" for name, errs in all_errors.items()
            )
            self.fail(f"Path errors found in config files:\n{error_msg}")

    def test_individual_config_files(self):
        """Test config files in config/individual directory"""
        individual_dir = CONFIG_DIR / "individual"
        if not individual_dir.exists():
            self.skipTest(f"Individual config directory not found: {individual_dir}")

        config_files = list(individual_dir.glob("*.yaml")) + list(individual_dir.glob("*.yml"))

        all_errors = {}
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                if config:
                    errors = validate_config_paths(config)
                    if errors:
                        all_errors[config_file.name] = errors
            except Exception as e:
                all_errors[config_file.name] = [f"Failed to parse: {e}"]

        if all_errors:
            error_msg = "\n".join(
                f"  {name}: {errs}" for name, errs in all_errors.items()
            )
            self.fail(f"Path errors found in individual config files:\n{error_msg}")


class TestPathJoinWithInvalidTypes(unittest.TestCase):
    """Test that Path operations fail with invalid types (documenting the bug)"""

    def test_path_join_with_bool_fails(self):
        """Document that Path / bool raises TypeError"""
        base = Path("/some/base")

        with self.assertRaises(TypeError) as ctx:
            _ = base / True

        self.assertIn("bool", str(ctx.exception).lower())

    def test_path_join_with_none_fails(self):
        """Document that Path / None raises TypeError"""
        base = Path("/some/base")

        with self.assertRaises(TypeError) as ctx:
            _ = base / None

        # Error message varies by Python version
        self.assertTrue(
            "NoneType" in str(ctx.exception) or "None" in str(ctx.exception)
        )

    def test_path_join_with_int_fails(self):
        """Document that Path / int raises TypeError"""
        base = Path("/some/base")

        with self.assertRaises(TypeError) as ctx:
            _ = base / 123

    def test_path_constructor_with_bool_fails(self):
        """Document that Path(bool) raises TypeError"""
        with self.assertRaises(TypeError):
            _ = Path(True)

    def test_path_constructor_with_none_returns_current(self):
        """Document that Path(None) actually works but returns '.'"""
        # Interestingly, Path(None) doesn't fail in some Python versions
        # but behaves unexpectedly
        try:
            p = Path(None)
            # If it doesn't fail, it returns current directory
            self.assertEqual(str(p), ".")
        except TypeError:
            # This is the expected behavior in most cases
            pass


class TestSafePathConstruction(unittest.TestCase):
    """Test safe path construction patterns"""

    def test_safe_path_get_with_default(self):
        """Test safe pattern for getting path from config"""
        config = {"paths": {"output_dir": None}}

        # Unsafe pattern (causes TypeError if value is bool/None and we use /):
        # path = base / config["paths"].get("output_dir", "default")

        # Safe pattern:
        def safe_get_path(config: dict, key: str, default: str) -> str:
            value = config.get(key)
            if value is None:
                return default
            if isinstance(value, bool):
                raise ValueError(f"{key} must be a path string, not bool")
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a path string, got {type(value).__name__}")
            return value

        # Test with None - returns default
        result = safe_get_path(config["paths"], "output_dir", "modelweights")
        self.assertEqual(result, "modelweights")

        # Test with valid string
        config["paths"]["output_dir"] = "custom/path"
        result = safe_get_path(config["paths"], "output_dir", "modelweights")
        self.assertEqual(result, "custom/path")

        # Test with bool - raises ValueError
        config["paths"]["output_dir"] = True
        with self.assertRaises(ValueError) as ctx:
            safe_get_path(config["paths"], "output_dir", "modelweights")
        self.assertIn("bool", str(ctx.exception))


class TestConfigValidationIntegration(unittest.TestCase):
    """Integration tests for config validation with temp files"""

    def setUp(self):
        """Create temp directory for test configs"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory"""
        shutil.rmtree(self.temp_dir)

    def test_create_and_validate_bad_config(self):
        """Test creating a config with bool paths and validating it"""
        bad_config = {
            "model": {
                "base_model": "test/model",
                "cache_dir": True  # Bug: should be string
            },
            "paths": {
                "output_dir": "modelweights",
                "cache_dir": False  # Bug: should be string
            }
        }

        config_path = Path(self.temp_dir) / "bad_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(bad_config, f)

        # Reload and validate
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)

        errors = validate_config_paths(loaded)
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("model.cache_dir" in e for e in errors))
        self.assertTrue(any("paths.cache_dir" in e for e in errors))

    def test_create_and_validate_good_config(self):
        """Test creating a valid config and validating it"""
        good_config = {
            "model": {
                "base_model": "test/model",
                "cache_dir": "../tmpcache"
            },
            "paths": {
                "output_dir": "modelweights",
                "cache_dir": "datasets/cache"
            },
            "training": {
                "strategies": [
                    {
                        "name": "test_strategy",
                        "dataset": "datasets/test.json",
                        "image_dir": "datasets/images"
                    }
                ]
            }
        }

        config_path = Path(self.temp_dir) / "good_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(good_config, f)

        # Reload and validate
        with open(config_path, 'r') as f:
            loaded = yaml.safe_load(f)

        errors = validate_config_paths(loaded)
        self.assertEqual(errors, [])


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPathValueValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigPathValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestYAMLPathParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestRealConfigFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestPathJoinWithInvalidTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestSafePathConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidationIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
