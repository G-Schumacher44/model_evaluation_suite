"""
🏆 Champion Model Validator

This script validates a production-ready model pulled from the MLflow Model Registry
using a final holdout dataset. It uses a YAML configuration file to define the model,
data paths, and reporting preferences.

Features:
- Resolves the correct model version from the registry (via version or alias/stage)
- Detects task type (classification vs regression) from the model pipeline
- Generates holdout evaluation metrics and validation-specific plots
- Tags the model version with a production-ready status in MLflow
- Optionally displays a dashboard if in notebook mode

Usage:

CLI:
    $ python src/model_eval_suite/validate_champion.py config/validation_config.yaml

Jupyter Notebook:
    from model_eval_suite.validate_champion import validate_and_display
    validate_and_display("config/validation_config.yaml")
"""
import yaml
import pandas as pd
import mlflow
import argparse
from mlflow.tracking import MlflowClient

# Import the new config model and the existing core evaluators
from .validation.val_config import ValidationConfig, create_suite_config_from_validation_config
from .classification.class_evaluator import orchestrate_model_evaluation as orchestrate_classification
from .regression.reg_evaluator import orchestrate_model_evaluation as orchestrate_regression
from .utils.export_utils import export_artifacts
from .validation.val_dashboard import display_validation_dashboard
from .validation import val_plots as validation_plots # Import the new validation-specific plots


def validate_and_display(config_path: str):
    """
    Loads a validation config file, runs the validation, and displays the dashboard.
    This is the recommended function for use in notebooks.
    """
    try:
        # Load YAML validation config from file
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Parse into strongly typed ValidationConfig
        validation_config = ValidationConfig(**config_dict)

        # Run the core validation logic
        results = run_validation(validation_config)

        # Optionally display dashboard in notebook
        if results and validation_config.notebook_mode:
            print("--- 📊 Rendering Validation Dashboard ---")
            display_validation_dashboard(results)
        elif not results:
            print("Validation did not produce any results to display.")

    except FileNotFoundError:
        print(f"🚨 Error: Configuration file not found at '{config_path}'")
    except Exception as e:
        print(f"🚨 An unexpected error occurred: {e}")

def run_validation(config: ValidationConfig) -> dict:
    """Orchestrates the end-to-end champion model validation and returns the results."""
    print(f"--- 🚀 Starting Champion Model Validation: {config.report_name} ---")
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()

    # --- Step 1: Resolve and Load Models & Data First ---
    try:
        # Resolve and load champion model
        model_name = config.model_source.name
        model_version_str = str(config.model_source.version)
        if not model_version_str.isdigit():
            latest_versions = client.get_latest_versions(name=model_name, stages=[model_version_str] if model_version_str != "latest" else None)
            if not latest_versions:
                raise ValueError(f"No model version found for name '{model_name}' and stage/alias '{model_version_str}'")
            model_version = latest_versions[0].version
            print(f"Resolved '{model_version_str}' to version {model_version} for model '{model_name}'")
        else:
            model_version = model_version_str
        
        champion_model_uri = f"models:/{model_name}/{model_version}"
        print(f"Loading model from: {champion_model_uri}")
        model = mlflow.sklearn.load_model(champion_model_uri)

        # Load holdout data
        print(f"Loading holdout data from: {config.holdout_data_path}")
        holdout_df = pd.read_csv(config.holdout_data_path)
        X_holdout = holdout_df.drop(columns=[config.target_column])
        y_holdout = holdout_df[config.target_column]

    except mlflow.exceptions.MlflowException as e:
        print(f"🚨 MLflow Error: Could not load model '{config.model_source.name}'. Details: {e}")
        return {}
    except FileNotFoundError:
        print(f"🚨 File Not Found Error: Could not find holdout data at '{config.holdout_data_path}'.")
        return {}
    except Exception as e:
        print(f"🚨 An unexpected error occurred during setup: {e}")
        return {}

    # --- Step 2: Evaluate Baseline Model (if specified) ---
    baseline_metrics = None
    suite_config = create_suite_config_from_validation_config(config)
    
    if config.baseline_model:
        try:
            baseline_uri = f"models:/{config.baseline_model.name}/{config.baseline_model.version}"
            print(f"Loading baseline model from: {baseline_uri}")
            baseline_model = mlflow.sklearn.load_model(baseline_uri)
            
            baseline_results = {}
            estimator_name = model.steps[-1][1].__class__.__name__
            task_type = "regression" if "Regressor" in estimator_name else "classification"
            suite_config.task_type = task_type

            print(f"Evaluating baseline model...")
            if task_type == "classification":
                orchestrate_classification(
                    model=baseline_model, X_train=X_holdout, y_train=y_holdout,
                    X_test=X_holdout, y_test=y_holdout, config=suite_config, results=baseline_results
                )
            else:
                orchestrate_regression(
                    model=baseline_model, X_train=X_holdout, y_train=y_holdout,
                    X_test=X_holdout, y_test=y_holdout, config=suite_config, results=baseline_results
                )
            baseline_metrics = baseline_results.get('metrics', {})
        except Exception as e:
            print(f"⚠️ Warning: Failed to load or evaluate baseline model: {e}")

    # --- Step 3: Evaluate Champion Model ---
    estimator_name = model.steps[-1][1].__class__.__name__
    task_type = "regression" if "Regressor" in estimator_name else "classification"
    suite_config.task_type = task_type
    print(f"Detected task type: {task_type}")

    results = {}
    if task_type == "classification":
        orchestrate_classification(
            model=model, X_train=X_holdout, y_train=y_holdout,
            X_test=X_holdout, y_test=y_holdout, config=suite_config, results=results
        )
    else:
        orchestrate_regression(
            model=model, X_train=X_holdout, y_train=y_holdout,
            X_test=X_holdout, y_test=y_holdout, config=suite_config, results=results
        )

    # Generate and Add Custom Validation Plots
    if results:
        print("Generating final assessment plots...")
        # Initialize container for validation plots
        results.setdefault('plot_paths', {})

        # Use an if/elif block to call the correct plots for the task type
        if task_type == 'classification':
            y_pred = model.predict(X_holdout)
            
            # Generate accuracy confidence interval plot for validation dashboard
            ci_plot_path = validation_plots.plot_accuracy_confidence_interval(y_holdout, y_pred, suite_config)
            results['plot_paths']['accuracy_confidence_interval'] = ci_plot_path
            
            if config.segmentation_columns:
                for column in config.segmentation_columns:
                    # Generate performance by segment plot for validation dashboard
                    segment_plot_path = validation_plots.plot_performance_by_segment(X_holdout, y_holdout, y_pred, column, suite_config)
                    results['plot_paths'][f'segment_performance_{column}'] = segment_plot_path

        elif task_type == 'regression':
            y_pred = model.predict(X_holdout)

            # Generate predicted vs actual with intervals plot for validation dashboard
            interval_plot_path = validation_plots.plot_predicted_vs_actual_with_intervals(model, X_holdout, y_holdout, suite_config)
            if interval_plot_path:
                 results['plot_paths']['pred_vs_actual_intervals'] = interval_plot_path

            if config.segmentation_columns:
                for column in config.segmentation_columns:
                    # Generate residuals by segment plot for validation dashboard
                    segment_plot_path = validation_plots.plot_residuals_by_segment(X_holdout, y_holdout, y_pred, column, suite_config)
                    results['plot_paths'][f'segment_residuals_{column}'] = segment_plot_path

    # Compute delta metrics if baseline metrics are available
    if baseline_metrics:
        results['baseline_metrics'] = baseline_metrics
        results['comparison_metrics'] = {}
        for key, value in results.get('metrics', {}).items():
            if key in baseline_metrics:
                try:
                    delta = value - baseline_metrics[key]
                    results['comparison_metrics'][key] = {
                        'value': value,
                        'baseline': baseline_metrics[key],
                        'delta': delta
                    }
                    # Optionally include stddevs if available in either set
                    if 'stddevs' in results and key in results['stddevs']:
                        results['comparison_metrics'][key]['std'] = results['stddevs'][key]
                    if 'stddevs' in baseline_results and key in baseline_results['stddevs']:
                        results['comparison_metrics'][key]['baseline_std'] = baseline_results['stddevs'][key]
                except:
                    pass  # Skip non-numeric metrics

    # Save metrics, plots, and artifacts to disk
    print("Exporting validation artifacts...")
    export_artifacts(results, model, suite_config)

    # Tag model in MLflow with validation status (e.g., "ready-for-prod")
    print(f"Tagging model version with status: '{config.production_tag}'")
    client.set_model_version_tag(name=model_name, version=model_version, key="status", value=config.production_tag)

    print(f"--- ✅ Validation Complete for {model_name} v{model_version} ---")

    return results


def main():
    """Main entry point for CLI execution."""
    # CLI entrypoint for running validation from terminal
    parser = argparse.ArgumentParser(description="Validate a champion model from the MLflow Registry.")
    parser.add_argument("config_path", type=str, help="Path to the validation YAML config file.")
    args = parser.parse_args()

    try:
        with open(args.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = ValidationConfig(**config_dict)
        run_validation(config)

    except FileNotFoundError:
        print(f"🚨 Error: Configuration file not found at '{args.config_path}'")
    except Exception as e:
        print("🚨 Error parsing validation config. Please check the YAML schema.")
        print(f"Details: {e}")


if __name__ == '__main__':
    main()