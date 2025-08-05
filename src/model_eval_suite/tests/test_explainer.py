# src/model_eval_suite/tests/test_explainer.py

# src/model_eval_suite/tests/test_explainer.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from model_eval_suite.utils.config import SuiteConfig, load_config
from model_eval_suite.modeling.explainers import generate_shap_explainer_and_values

def test_shap_explainer_runs():
    df = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 1, 0, 0]})
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
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
            "test_data_path": "test.csv"
        },
        modeling={
            "target_column": "target",
            "pipeline_factory": {
                "name": "LogisticRegression",
                "params": {},
                "numeric_features": ["f1", "f2"],
                "categorical_features": []
            }
        },
        evaluation={
            "run": True,
            "export_xlsx_summary": False,
            "export_html_dashboard": False,
            "compare_to_baseline": None,
            "plots": {},
            "explainability": {"run_shap": True},
            "audits": {}
        }
    )
    result = generate_shap_explainer_and_values(model, df, dummy_config)
    assert "shap_values" in result
    assert result["shap_values"].values.shape[0] == df.shape[0]