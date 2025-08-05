# src/model_eval_suite/tests/test_class_plots.py

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from model_eval_suite.classification.class_plots import generate_all_plots
from model_eval_suite.utils.config import SuiteConfig, PathsConfig, EvaluationConfig, PlotConfig, ModelingConfig, PipelineFactoryConfig

@pytest.fixture
def dummy_config(tmp_path):
    return SuiteConfig(
        run_id="plot_test_run",
        task_type="classification",
        notebook_mode=False,
        logging="auto",
        paths=PathsConfig(
            input_data=tmp_path / "fake.csv",
            reports_dir=tmp_path / "reports",
            plots_dir=tmp_path / "plots",
            model_export_dir=tmp_path / "models",
            metrics_log=tmp_path / "metrics.csv",
            log_dir=tmp_path / "logs",
            train_data_path=tmp_path / "train.csv",
            test_data_path=tmp_path / "test.csv"
        ),
        modeling=ModelingConfig(
            target_column="label",
            pipeline_factory=PipelineFactoryConfig(name="LogisticRegression")
        ),
        evaluation=EvaluationConfig(
            run=True,
            export_xlsx_summary=False,
            export_html_dashboard=False,
            plots=PlotConfig()
        )
    )

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "cat_feature": np.random.choice(["A", "B", "C"], 100)
    })
    y = np.random.choice([0, 1], 100)
    return X, y

@pytest.fixture
def dummy_model(dummy_data):
    X, y = dummy_data
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["feature1", "feature2"])
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("estimator", LogisticRegression())
    ])
    model.fit(X[["feature1", "feature2"]], y)
    return model

def test_generate_all_plots_runs(dummy_model, dummy_data, dummy_config):
    X, y = dummy_data
    results = {"metrics": {"raw_cv_scores": [0.82, 0.79, 0.80, 0.78, 0.81]}}
    plot_paths = generate_all_plots(
        model=dummy_model,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        config=dummy_config,
        results=results
    )
    assert isinstance(plot_paths, dict)
    assert all(isinstance(p, str) for p in plot_paths.values())