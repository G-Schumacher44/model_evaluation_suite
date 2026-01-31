"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
import yaml

from model_eval_suite.utils.config import SuiteConfig, load_config


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_load_config_from_dict(self, sample_config_dict, temp_dir):
        """Test loading config from dictionary."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = load_config(str(config_path))
        assert isinstance(config, SuiteConfig)
        assert config.run_id == "test_run_01"
        assert config.task_type == "classification"

    def test_config_validation_missing_required_fields(self, temp_dir):
        """Test that config validation fails with missing required fields."""
        invalid_config = {
            "run_id": "test",
            # Missing task_type and other required fields
        }
        config_path = temp_dir / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        with pytest.raises(ValueError):  # Pydantic ValidationError raises ValueError
            load_config(str(config_path))

    def test_config_paths_conversion(self, sample_config_dict, temp_dir):
        """Test that paths are converted to Path objects."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = load_config(str(config_path))
        assert isinstance(config.paths.input_data, Path)
        assert isinstance(config.paths.reports_dir, Path)
        assert isinstance(config.paths.log_dir, Path)

    def test_config_default_values(self, sample_config_dict, temp_dir):
        """Test that default values are applied."""
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = load_config(str(config_path))
        assert not config.notebook_mode
        assert config.logging == "off"
        assert config.mlflow_tracking_uri == "sqlite:///test_mlflow.db"


class TestSuiteConfig:
    """Test SuiteConfig model validation."""

    def test_task_type_validation(self, sample_config_dict):
        """Test that task_type accepts valid values."""
        # Test valid values
        for valid_type in ["classification", "regression"]:
            sample_config_dict["task_type"] = valid_type
            config = SuiteConfig(**sample_config_dict)
            assert config.task_type == valid_type

    def test_mlflow_tracking_uri_default(self, sample_config_dict):
        """Test that mlflow_tracking_uri has correct default."""
        del sample_config_dict["mlflow_tracking_uri"]
        config = SuiteConfig(**sample_config_dict)
        assert config.mlflow_tracking_uri == "sqlite:///mlflow.db"
