# src/model_eval_suite/utils/config.py


from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Any
from pathlib import Path
import yaml

"""
⚙️ Configuration Schema for Model Evaluation Suite

This module defines the Pydantic-based configuration schema used to load and validate
all runtime settings from a YAML config file.

Responsibilities:
- Define structured config blocks for paths, evaluation, modeling, diagnostics, and feature engineering
- Support runtime toggles for dashboard export, SHAP explainability, plot generation, etc.
- Enforce types and default values for deeply nested configuration objects

Entrypoint:
- load_config(): Loads and parses the YAML file into a SuiteConfig object
"""

 # File paths used throughout the evaluation workflow
class PathsConfig(BaseModel):
    input_data: Path      
    reports_dir: Path     
    plots_dir: Path       
    model_export_dir: Path
    metrics_log: Path
    log_dir: Path 
    train_data_path: Path 
    test_data_path: Path
       

 # Flags for controlling which plots are generated and saved
class PlotConfig(BaseModel):
    confusion_matrix: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    roc_curve: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    pr_curve: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    learning_curve: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    calibration_curve: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    threshold_plot: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    permutation_importance: Dict[str, Any] = Field(default_factory=lambda: {'save': True, 'plot_type': 'box'})
    lift_curve: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    cumulative_gain: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    shap_summary: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    feature_distributions: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    predicted_vs_actual: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    residuals_plot: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    residuals_histogram: Dict[str, Any] = Field(default_factory=lambda: {'save': True})
    qq_plot: Dict[str, Any] = Field(default_factory=lambda: {'save': True})

# Settings for Audit Alerts
class AuditConfig(BaseModel):
    """Settings for Audit Alerts."""
    # Regression thresholds
    r_squared_min_threshold: float = 0.7
    rmse_max_threshold: float = 10000.0
    performance_regression_r2_threshold_factor: float = 0.95

    # Classification thresholds
    overfitting_threshold_factor: float = 1.15
    cv_std_threshold: float = 0.1
    performance_regression_f1_threshold_factor: float = 0.95

 # Settings for evaluation outputs and visualizations
class EvaluationConfig(BaseModel):
    run: bool = True
    export_xlsx_summary: bool = False
    export_html_dashboard: bool = True  # Added new field
    compare_to_baseline: Optional[str] = None
    plots: PlotConfig = Field(default_factory=PlotConfig)
    explainability: Optional[Dict[str, Any]] = None
    audits: AuditConfig = Field(default_factory=AuditConfig)
    
 # Configuration for optional grid search tuning
class HyperparameterTuningConfig(BaseModel):
    run: bool = False
    param_grid: Dict[str, Any]
    cv_folds: int = 5
    scoring: str = 'f1'
    verbose: int = 1

 # Optional user-defined feature engineering module
class FeatureEngineeringConfig(BaseModel):
    run: bool = False
    module: str
    class_name: str

 # Specifies how to build the model pipeline
class PipelineFactoryConfig(BaseModel):
    name: str
    registered_name: Optional[str] = None #<-- ADD THIS LINE
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    params: Dict[str, Any] = {}

 # Core modeling settings including target and pipeline details
class ModelingConfig(BaseModel):
    target_column: str
    test_size: float = 0.2
    feature_engineering: Optional[FeatureEngineeringConfig] = None
    hyperparameter_tuning: Optional[HyperparameterTuningConfig] = None
    pipeline_factory: PipelineFactoryConfig

 # Optional pre-modeling data diagnostics configuration
class PreModelDiagnosticsConfig(BaseModel):
    run: bool = False
    export_reports: bool = True
    export_html_report: bool = True
    vif_threshold: float = 5.0
    skewness_threshold: float = 0.75

 # Top-level schema encompassing the full configuration file
class SuiteConfig(BaseModel):
    run_id: str
    task_type: str
    pre_model_diagnostics: Optional[PreModelDiagnosticsConfig] = None # <-- ADD THIS LINE
    notebook_mode: bool = True
    logging: str = "auto"
    paths: PathsConfig
    modeling: ModelingConfig
    evaluation: EvaluationConfig

 # Load and parse YAML config into structured SuiteConfig object
def load_config(path: str) -> SuiteConfig:
    """Loads a YAML file and parses it into a SuiteConfig object."""
    with open(path, "r") as f:
        raw_config = yaml.safe_load(f)
    return SuiteConfig(**raw_config)