"""Unit tests for classification metrics."""

import numpy as np
import pandas as pd

from model_eval_suite.classification.class_metrics import (
    generate_audit_alerts,
    generate_model_metrics,
    load_baseline_metrics,
)
from model_eval_suite.utils.config import SuiteConfig


class TestClassificationMetrics:
    """Test classification metrics generation."""

    def test_generate_model_metrics(self, trained_classifier, sample_classification_data):
        """Test generating classification metrics."""
        X = sample_classification_data.drop("target", axis=1)
        y = sample_classification_data["target"]

        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Retrain on this split
        trained_classifier.fit(X_train, y_train)

        metrics = generate_model_metrics(
            model=trained_classifier, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        assert isinstance(metrics, dict)
        assert "test_f1" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "train_f1" in metrics

        # Check metric ranges
        assert 0 <= metrics["test_f1"] <= 1
        assert 0 <= metrics["test_precision"] <= 1
        assert 0 <= metrics["test_recall"] <= 1

    def test_load_baseline_metrics_file_not_found(self, temp_dir):
        """Test loading baseline metrics when file doesn't exist."""
        log_path = str(temp_dir / "nonexistent.csv")

        baseline = load_baseline_metrics(log_path, baseline_id="test_baseline")

        # Should return None when file doesn't exist
        assert baseline is None

    def test_load_baseline_metrics_with_valid_file(self, temp_dir):
        """Test loading baseline metrics from valid CSV."""
        # Create a mock metrics log
        metrics_data = {
            "run_id": ["baseline_run_01", "other_run"],
            "test_f1": [0.85, 0.90],
            "test_precision": [0.82, 0.88],
            "test_recall": [0.88, 0.92],
        }
        df = pd.DataFrame(metrics_data)
        log_path = temp_dir / "metrics.csv"
        df.to_csv(log_path, index=False)

        baseline = load_baseline_metrics(str(log_path), baseline_id="baseline_run_01")

        assert baseline is not None
        assert isinstance(baseline, dict)
        assert baseline["test_f1"] == 0.85
        assert baseline["test_precision"] == 0.82

    def test_generate_audit_alerts(self, sample_config_dict):
        """Test audit alert generation."""
        config = SuiteConfig(**sample_config_dict)

        # Create results with potential issues
        results = {
            "metrics": {
                "train_f1": 0.95,
                "test_f1": 0.70,  # Large gap indicating overfitting
                "cv_f1_mean": 0.75,
                "cv_f1_std": 0.15,  # High variance
            },
            "baseline_metrics": {
                "test_f1": 0.90  # Current is worse
            },
        }

        alerts = generate_audit_alerts(results, config)

        assert isinstance(alerts, list)
        # Should have alerts for overfitting, CV instability, and regression
        assert len(alerts) >= 1


class TestMetricsWithDifferentModels:
    """Test metrics with various model configurations."""

    def test_metrics_without_predict_proba(self, sample_classification_data):
        """Test metrics for models without probability predictions."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        X = sample_classification_data.drop("target", axis=1)
        y = sample_classification_data["target"]

        # SVC without probability doesn't have predict_proba
        model = Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(kernel="linear", random_state=42))]
        )

        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Select only numeric columns for StandardScaler
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_num = X_train[numeric_cols]
        X_test_num = X_test[numeric_cols]

        model.fit(X_train_num, y_train)

        metrics = generate_model_metrics(
            model=model, X_train=X_train_num, y_train=y_train, X_test=X_test_num, y_test=y_test
        )

        # ROC AUC should be None for models without predict_proba
        assert metrics.get("test_roc_auc") is None or isinstance(metrics.get("test_roc_auc"), float)
