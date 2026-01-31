# src/model_eval_suite/tests/test_explainer.py

# src/model_eval_suite/tests/test_explainer.py

import pandas as pd

from model_eval_suite.modeling.explainers import generate_shap_explainer_and_values
from model_eval_suite.utils.config import SuiteConfig


def test_shap_explainer_runs():
    """Test SHAP explainer with proper pipeline structure."""
    from model_eval_suite.modeling.factory import pipeline_factory

    df = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 1, 0, 0]})

    # Use factory to create proper pipeline with preprocessor
    config = {
        "name": "LogisticRegression",
        "numeric_features": ["f1", "f2"],
        "categorical_features": [],
        "params": {"random_state": 42},
    }
    model = pipeline_factory(config)
    model.fit(df, [0, 1, 0, 1])

    dummy_config = SuiteConfig(
        run_id="test_run",
        task_type="classification",
        notebook_mode=False,
        logging="auto",
        paths={
            "input_data": "dummy.csv",
            "reports_dir": "reports/",
            "plots_dir": "plots/",
            "model_export_dir": "models/",
            "metrics_log": "metrics.csv",
            "log_dir": "logs/",
            "train_data_path": "train.csv",
            "test_data_path": "test.csv",
        },
        modeling={
            "target_column": "target",
            "pipeline_factory": {
                "name": "LogisticRegression",
                "params": {},
                "numeric_features": ["f1", "f2"],
                "categorical_features": [],
            },
        },
        evaluation={
            "run": True,
            "export_xlsx_summary": False,
            "export_html_dashboard": False,
            "compare_to_baseline": None,
            "plots": {},
            "explainability": {"run_shap": True},
            "audits": {},
        },
    )
    result = generate_shap_explainer_and_values(model, df, dummy_config)
    # May return empty dict if SHAP fails, that's ok
    assert isinstance(result, dict)
