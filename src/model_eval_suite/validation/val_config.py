# src/model_eval_suite/validation/config.py
from pydantic import BaseModel
from pathlib import Path
from typing import Union, Dict, Any, Optional, List

"""
✅ Validation Config for Champion Model Evaluation

This module defines the configuration schema used for post-modeling validation of a
champion model selected from MLflow or other sources.

Responsibilities:
- Define how to locate the production-ready model (via ModelSourceConfig)
- Specify the holdout dataset and target column
- Identify model type, segmentation, and feature schema
- Automatically construct a compatible SuiteConfig for reuse across the evaluation suite

Entrypoint:
- create_suite_config_from_validation_config(): Converts ValidationConfig into a full SuiteConfig
"""

# Import the main suite's config models to build upon them
from ..utils.config import SuiteConfig, PathsConfig, ModelingConfig, EvaluationConfig, PipelineFactoryConfig

class ModelSourceConfig(BaseModel):
    """Specifies how to find the champion model."""
    name: str
    version: Union[int, str] = "latest"

class ValidationConfig(BaseModel):
    """This class must use model_source."""
    mlflow_tracking_uri: str
    # Model registry reference
    model_source: ModelSourceConfig
    baseline_model: Optional[ModelSourceConfig] = None
    # Holdout dataset
    holdout_data_path: Path
    target_column: str
    report_name: str
    production_tag: str = "production"
    notebook_mode: bool = True
    task_type: str = "classification"
    segmentation_columns: Optional[List[str]] = None
    # Input features
    numeric_features: List[str]
    categorical_features: List[str]

def create_suite_config_from_validation_config(config: ValidationConfig) -> SuiteConfig:
    """
    Creates a SuiteConfig object from a ValidationConfig.
    This version correctly enables plot and explainer generation.
    """
    run_id = config.report_name
    
    paths = PathsConfig(
        input_data=config.holdout_data_path, reports_dir=Path("exports/reports"),
        plots_dir=Path("exports/plots"), model_export_dir=Path("models"),
        metrics_log=Path("exports/reports/model_metrics_log.csv"), log_dir=Path("logs"),
        train_data_path=config.holdout_data_path, test_data_path=config.holdout_data_path,
    )
    
    
    # Handle baseline model metadata if available
    if config.baseline_model:
        baseline_metadata = {
            "registered_name": config.baseline_model.name,
            "version": config.baseline_model.version
        }

    # explainability and all plots by default for the validation run.
    evaluation_config = EvaluationConfig(
        run=True,
        export_html_dashboard=True,
        explainability={"run_shap": True, "shap_sample_size": 2000},
        baseline_metadata=baseline_metadata
    )
    
    suite_config = SuiteConfig(
        run_id=config.report_name,
        task_type=config.task_type,
        notebook_mode=config.notebook_mode,
        paths=paths, 
        modeling=ModelingConfig(
            target_column=config.target_column,
            # pass the feature lists
            pipeline_factory=PipelineFactoryConfig(
                name="loaded_from_registry",
                numeric_features=config.numeric_features,
                categorical_features=config.categorical_features
            )
        ),
        evaluation=evaluation_config 
    )
    
    return suite_config