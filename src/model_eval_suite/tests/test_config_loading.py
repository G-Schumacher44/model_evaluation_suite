import yaml
from model_eval_suite.utils.config import load_config


def test_config_loads_successfully(tmp_path):
    """
    Ensures the load_config function correctly loads and validates a minimal SuiteConfig.
    """
    config_dict = {
        "run_id": "test_run",
        "task_type": "classification",
        "notebook_mode": False,
        "logging": "auto",
        "paths": {
            "input_data": "data/input.csv",
            "reports_dir": "data/reports",
            "plots_dir": "data/plots",
            "model_export_dir": "data/models",
            "metrics_log": "data/logs/metrics_log.csv",
            "log_dir": "data/logs",
            "train_data_path": "data/train.csv",
            "test_data_path": "data/test.csv"
        },
        "modeling": {
            "target_column": "churn",
            "pipeline_factory": {
                "name": "DummyClassifier",
                "numeric_features": [],
                "categorical_features": [],
                "params": {}
            }
        },
        "evaluation": {
            "run": True,
            "export_xlsx_summary": False,
            "export_html_dashboard": True,
            "plots": {}
        }
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f)

    config = load_config(str(config_path))
    assert config.run_id == "test_run"
    assert config.task_type == "classification"
    assert config.paths.input_data.name == "input.csv"
