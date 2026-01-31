"""Unit tests for plot generation validation."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class TestPlotGeneration:
    """Test plot generation utilities."""

    def test_matplotlib_figure_creation(self):
        """Test basic matplotlib figure creation."""
        fig, ax = plt.subplots(figsize=(8, 6))
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_with_data(self, sample_classification_data):
        """Test creating a simple plot with data."""
        data = sample_classification_data["feature_1"]

        fig, ax = plt.subplots()
        ax.hist(data, bins=10)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Frequency")

        assert len(ax.patches) > 0  # Should have bars
        plt.close(fig)

    def test_scatter_plot_creation(self, sample_regression_data):
        """Test scatter plot creation."""
        X = sample_regression_data["feature_1"]
        y = sample_regression_data["target"]

        fig, ax = plt.subplots()
        ax.scatter(X, y, alpha=0.5)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")

        # Check that plot has data
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_multiple_subplots(self):
        """Test creating multiple subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        assert axes.shape == (2, 2)
        assert len(axes.flatten()) == 4

        plt.close(fig)

    def test_confusion_matrix_shape(self, sample_classification_data):
        """Test confusion matrix data structure."""
        from sklearn.metrics import confusion_matrix

        y_true = sample_classification_data["target"]
        y_pred = np.random.randint(0, 2, size=len(y_true))

        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)  # Binary classification
        assert cm.sum() == len(y_true)


class TestPlotSaving:
    """Test plot saving functionality."""

    def test_save_figure_to_file(self, temp_dir):
        """Test saving a plot to file."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        output_path = temp_dir / "test_plot.png"
        fig.savefig(output_path, dpi=100, bbox_inches="tight")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        plt.close(fig)

    def test_save_multiple_formats(self, temp_dir):
        """Test saving plots in different formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        formats = ["png", "pdf", "svg"]

        for fmt in formats:
            output_path = temp_dir / f"plot.{fmt}"
            fig.savefig(output_path)
            assert output_path.exists()

        plt.close(fig)
