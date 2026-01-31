"""Pytest configuration and shared fixtures for model_eval_suite tests."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_classification_data():
    """Generate sample binary classification dataset."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.randint(0, 5, n_samples),
        "cat_feature": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 100

    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    y = 3 * X1 + 2 * X2 + np.random.randn(n_samples) * 0.1

    data = {
        "feature_1": X1,
        "feature_2": X2,
        "cat_feature": np.random.choice(["A", "B"], n_samples),
        "target": y,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_config_dict():
    """Minimal valid configuration dictionary."""
    return {
        "run_id": "test_run_01",
        "task_type": "classification",
        "notebook_mode": False,
        "logging": "off",
        "mlflow_tracking_uri": "sqlite:///test_mlflow.db",
        "paths": {
            "input_data": "data/input.csv",
            "reports_dir": "reports",
            "plots_dir": "plots",
            "model_export_dir": "models",
            "metrics_log": "logs/metrics.csv",
            "log_dir": "logs",
            "train_data_path": "data/train.csv",
            "test_data_path": "data/test.csv",
        },
        "modeling": {
            "target_column": "target",
            "pipeline_factory": {
                "name": "LogisticRegression",
                "numeric_features": ["feature_1", "feature_2"],
                "categorical_features": ["cat_feature"],
                "params": {"random_state": 42},
            },
            "feature_engineering": None,
            "hyperparameter_tuning": None,
        },
        "evaluation": {
            "run": True,
            "export_xlsx_summary": False,
            "export_html_dashboard": False,
            "plots": {},
        },
    }


@pytest.fixture
def trained_classifier(sample_classification_data):
    """Return a trained classifier pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X = sample_classification_data.drop("target", axis=1)
    y = sample_classification_data["target"]

    numeric_features = ["feature_1", "feature_2", "feature_3"]
    categorical_features = ["cat_feature"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", LogisticRegression(random_state=42))]
    )

    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def trained_regressor(sample_regression_data):
    """Return a trained regressor pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X = sample_regression_data.drop("target", axis=1)
    y = sample_regression_data["target"]

    numeric_features = ["feature_1", "feature_2"]
    categorical_features = ["cat_feature"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])

    pipeline.fit(X, y)
    return pipeline
