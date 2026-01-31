"""Unit tests for export utilities."""

from pathlib import Path

import pandas as pd
import pytest


class TestExportUtilities:
    """Test export utility functions."""

    def test_dataframe_export_to_csv(self, sample_classification_data, temp_dir):
        """Test exporting DataFrame to CSV."""
        output_path = temp_dir / "test_export.csv"

        sample_classification_data.to_csv(output_path, index=False)

        assert output_path.exists()

        # Verify content
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == len(sample_classification_data)
        assert list(df_loaded.columns) == list(sample_classification_data.columns)

    def test_create_export_directory(self, temp_dir):
        """Test creating export directories."""
        export_dir = temp_dir / "exports" / "test_run"

        export_dir.mkdir(parents=True, exist_ok=True)

        assert export_dir.exists()
        assert export_dir.is_dir()

    def test_path_handling_with_string(self, temp_dir):
        """Test path handling with string input."""
        path_str = str(temp_dir / "test.csv")
        path_obj = Path(path_str)

        assert path_obj.parent == temp_dir
        assert path_obj.name == "test.csv"

    def test_metrics_dict_to_csv(self, temp_dir):
        """Test exporting metrics dictionary to CSV."""
        metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88, "f1_score": 0.85}

        df = pd.DataFrame([metrics])
        output_path = temp_dir / "metrics.csv"
        df.to_csv(output_path, index=False)

        assert output_path.exists()

        # Verify
        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) == 1
        assert df_loaded["accuracy"].iloc[0] == pytest.approx(0.85)
