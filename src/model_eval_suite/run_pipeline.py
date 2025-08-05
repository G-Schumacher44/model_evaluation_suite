"""
🏃 Model Evaluation Runner

This script serves as the main orchestrator for a model evaluation run. It merges a
default YAML configuration with an optional user-provided override file, then executes
the full pipeline including diagnostics, training, evaluation, and export.

Key Functions:
- Loads data and applies pre-modeling diagnostics
- Constructs and fits pipelines via the factory pattern
- Optionally performs GridSearchCV hyperparameter tuning
- Logs model and metrics to MLflow
- Renders dashboards (if in notebook mode)

Usage:

CLI:
    $ python src/model_eval_suite/runner.py config/my_override_config.yaml

Jupyter Notebook:
    from model_eval_suite.runner import main
    main(user_config_path="config/my_override_config.yaml")
"""
import os
import sys
import yaml
import json
import pandas as pd
import mlflow
import argparse
import logging
from typing import Any, Dict, Optional
from collections.abc import Mapping
import importlib.resources
from pydantic import ValidationError

# Import Block
from model_eval_suite.utils.config import SuiteConfig
from model_eval_suite.utils.logging import configure_logging
from model_eval_suite.utils.export_utils import export_artifacts
from model_eval_suite.modeling.factory import pipeline_factory
from mlflow.models import infer_signature
from model_eval_suite.utils.pre_model_diagnostics import run_pre_model_diagnostics
from model_eval_suite.utils.premodeling_dashboard import display_pre_modeling_dashboard
from sklearn.model_selection import train_test_split, GridSearchCV

def deep_merge(d1: Dict, d2: Dict) -> Dict:
    """Recursively merges dictionary d2 into d1."""
    for k, v in d2.items():
        if isinstance(v, Mapping):
            d1[k] = deep_merge(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1

def main(user_config_path: Optional[str] = None):
    """
    Main function to orchestrate an evaluation run by merging a default
    config with a user-provided override config.
    """
    # 1. Load base/default and optional override configs
    with importlib.resources.files('model_eval_suite.config').joinpath('default_config.yaml').open('r') as f:
        config_dict = yaml.safe_load(f)

    if user_config_path:
        with open(user_config_path, "r") as f:
            user_config_dict = yaml.safe_load(f)
        config_dict = deep_merge(config_dict, user_config_dict)

    # 2. Extract specific run configuration from the merged config
    run_name = config_dict.get("run_to_execute")
    if not run_name:
        raise ValueError("'run_to_execute' key not found in the final configuration.")
    
    run_config_dict = config_dict.get(run_name)
    if not run_config_dict:
        raise ValueError(f"Run name '{run_name}' not found in the final configuration.")

    final_run_config = config_dict.get('base_config', {})
    final_run_config = deep_merge(final_run_config, run_config_dict)
    for key, value in config_dict.items():
        if key not in ['base_config'] and not isinstance(value, dict):
            final_run_config[key] = value

    # 3. Convert dict to strongly typed config object
    try:
        config = SuiteConfig(**final_run_config)
    except ValidationError as e:
        print("❌ Config validation failed:\n")
        print(e)
        sys.exit(1)

    # 4. Configure logging with log file path
    log_file_path = config.paths.log_dir / config.run_id / f"{config.run_id}_run.log"
    configure_logging(config.notebook_mode, config.logging, log_file_path)

    # 5. Start MLflow experiment and log full config
    mlflow.set_experiment(config.run_id.split('_')[0])

    with mlflow.start_run(run_name=config.run_id):
        mlflow.log_params(pd.json_normalize(config.model_dump(), sep='_').to_dict(orient='records')[0])
        # Initialize results dictionary for storing metrics and artifacts
        results = {}
        
         
        # 6. Load original input data for diagnostics
        logging.info(f"Loading data from: {config.paths.input_data}")
        df = pd.read_csv(config.paths.input_data)

        # 7. Optionally run pre-model diagnostics if enabled
        diagnostic_results = {}
        if config.pre_model_diagnostics and config.pre_model_diagnostics.run:
                logging.info("--- Running Pre-Modeling Diagnostics ---")
                diagnostic_results.update(run_pre_model_diagnostics(df.copy(), config=config.model_dump()))
                logging.info("--- Diagnostics Complete ---")

        # 8. Load prepared training and testing datasets
        logging.info(f"Loading train data from: {config.paths.train_data_path}")
        train_df = pd.read_csv(config.paths.train_data_path)
        
        logging.info(f"Loading test data from: {config.paths.test_data_path}")
        test_df = pd.read_csv(config.paths.test_data_path)
        
        X_train = train_df.drop(columns=[config.modeling.target_column])
        y_train = train_df[config.modeling.target_column]
        X_test = test_df.drop(columns=[config.modeling.target_column])
        y_test = test_df[config.modeling.target_column]
               
        # 9. Build pipeline using factory method
        pipeline = pipeline_factory(
            factory_config=config.modeling.pipeline_factory.model_dump(),
            fe_config=config.modeling.feature_engineering.model_dump() if config.modeling.feature_engineering else None
        )
        
        tuning_config = config.modeling.hyperparameter_tuning
        final_model: Any

        # 10. Fit model (with optional hyperparameter tuning)
        if tuning_config and tuning_config.run:
            from sklearn.model_selection import StratifiedKFold

            # Define CV strategy
            cv_strategy = None
            if config.task_type == 'classification':
                cv_strategy = StratifiedKFold(n_splits=tuning_config.cv_folds)
                logging.info("Using StratifiedKFold for cross-validation.")
            else:
                cv_strategy = tuning_config.cv_folds

            logging.info("Executing GridSearchCV for hyperparameter tuning...")
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=tuning_config.param_grid,
                cv=cv_strategy,
                scoring=tuning_config.scoring,
                n_jobs=-1,
                verbose=tuning_config.verbose
            )
            grid_search.fit(X_train, y_train)
            final_model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)
            
            cv_results = grid_search.cv_results_
            fold_keys = [f'split{i}_test_score' for i in range(tuning_config.cv_folds)]
            if all(key in cv_results for key in fold_keys):
                raw_cv_scores = [cv_results[key][grid_search.best_index_] for key in fold_keys]
                results.setdefault('metrics', {})['raw_cv_scores'] = raw_cv_scores
            
        else:
            logging.info("Fitting model with specified parameters...")
            final_model = pipeline
            final_model.fit(X_train, y_train)
        
        # 11. Log trained model with input/output signature to MLflow
        logging.info("Logging model to MLflow...")
        input_example = X_train.head()
        signature = infer_signature(input_example, final_model.predict(input_example))
        model_registry_name = config.modeling.pipeline_factory.registered_name or config.modeling.pipeline_factory.name

        mlflow.sklearn.log_model(
            sk_model=final_model, 
            registered_model_name=model_registry_name,
            signature=signature,
            input_example=input_example
        )
        
        # 12. Run evaluation and collect metrics
        if config.evaluation.run:
            if config.task_type == "classification":
                from model_eval_suite.classification.class_evaluator import orchestrate_model_evaluation
                orchestrate_model_evaluation(
                    final_model, X_train, y_train, X_test, y_test, config, results=results
                )
            elif config.task_type == "regression":
                from model_eval_suite.regression.reg_evaluator import orchestrate_model_evaluation as orchestrate_regression_evaluation
                orchestrate_regression_evaluation(
                    final_model, X_train, y_train, X_test, y_test, config, results=results
                )
            else:
                raise ValueError(f"Unknown task_type: {config.task_type}")
            
            results['final_model'] = final_model

            # --- Final Run Summary ---
            summary_msg = (
                f"\n✅ Run complete: `{config.run_id}`\n"
                f"📁 Artifacts saved to:\n"
                f"   - Plots:   {os.path.join(config.paths.plots_dir, config.run_id)}\n"
                f"   - Reports: {os.path.join(config.paths.reports_dir, config.run_id)}\n"
                f"📦 MLflow model: `{model_registry_name}`\n"
            )

            metrics_to_log = {k: v for k, v in results.get('metrics', {}).items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics_to_log)
            
            # 13. Save plots, metrics, and other artifacts
            logging.info("Exporting local artifacts...")
            export_artifacts(results, final_model, config)
            
            plot_dir = os.path.join(config.paths.plots_dir, config.run_id)
            report_dir = os.path.join(config.paths.reports_dir, config.run_id)
            if os.path.isdir(plot_dir):
                mlflow.log_artifacts(plot_dir, artifact_path="plots")
            if os.path.isdir(report_dir):
                mlflow.log_artifacts(report_dir, artifact_path="reports")

            logging.info(f"Artifacts exported and logged for run_id: {config.run_id}")
            
            # 14. If in notebook mode, display interactive dashboards
            if config.notebook_mode:
                print(summary_msg)
                print("--- Rendering Dashboards ---")
                if diagnostic_results:
                    display_pre_modeling_dashboard(diagnostic_results, config)
                
                # 2. Display the main model evaluation dashboard
                try:
                    if config.task_type == "classification":
                        from model_eval_suite.classification.class_dashboard import display_evaluation_dashboard
                        display_evaluation_dashboard(results)
                    elif config.task_type == "regression":
                        from model_eval_suite.regression.reg_dashboard import display_regression_dashboard
                        display_regression_dashboard(results)
                except Exception as e:
                    print(f"⚠️ Dashboard failed to render cleanly: {e}")

    if not config.notebook_mode:
        logging.info(summary_msg)

# CLI entrypoint: accept optional config override path
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model evaluation suite.")
    parser.add_argument("user_config_path", nargs='?', default=None, help="Optional: Path to the user's override config.yaml file.")
    args = parser.parse_args()
    main(user_config_path=args.user_config_path)