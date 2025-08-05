import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Optional

from .reg_metrics import generate_audit_alerts
from .reg_metrics import load_baseline_metrics
from model_eval_suite.utils.config import SuiteConfig
from . import reg_metrics, reg_plots
from model_eval_suite.modeling import explainers

"""
📉 Regression Evaluator for the model evaluation suite.

This module orchestrates the end-to-end evaluation process for regression models.

Responsibilities:
- Computes train/test regression metrics (R², RMSE, MAE, etc.)
- Generates visualizations such as residual plots, learning curves, and SHAP explainers
- Runs audit checks for underfitting, instability, or poor performance
- Optionally includes a statsmodels summary report for linear models
- Aggregates all outputs into a results dictionary used by the dashboard

Entrypoint:
- orchestrate_model_evaluation(): performs the full evaluation and returns results
"""


def orchestrate_model_evaluation(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig,
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """Orchestrates the entire evaluation process for a regression model."""
    # Ensure results dictionary is provided by the caller
    if results is None:
        raise ValueError("Results dictionary must be provided by the caller.")
    results["config_obj"] = config
    results['final_model'] = model
    
    # Generate regression performance metrics
    results.setdefault('metrics', {}).update(
        reg_metrics.generate_regression_metrics(model, X_test, y_test)
    )
    # Predict on test set
    results['y_pred'] = model.predict(X_test)

    # Generate SHAP explainers if configured
    explainer_data = explainers.generate_all_explainers(model, X_train, config)
    results.update(explainer_data)
    
    # Generate evaluation plots
    plot_paths = reg_plots.generate_all_plots(model, X_train, y_train, X_test, y_test, config, results)
    results['plot_paths'] = plot_paths
    
    # Optional: add statsmodels OLS summary for linear models
    if isinstance(model.named_steps['estimator'], LinearRegression):
        try:
            import statsmodels.api as sm
            # Extract preprocessing steps
            transformer_pipeline = model[:-1]
            # Transform training features
            X_train_transformed = transformer_pipeline.transform(X_train)
            # Get transformed feature names
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
            # Add constant for intercept
            X_train_with_const = sm.add_constant(X_train_transformed_df, has_constant='add')
            # Fit OLS model
            ols_model = sm.OLS(y_train, X_train_with_const).fit()
            results['statsmodels_summary_obj'] = ols_model
            
            # Save summary report to disk
            report_path = config.paths.reports_dir / config.run_id / "statistical_summary.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(ols_model.summary().as_text())
            results.setdefault('plot_paths', {})['statistical_summary'] = str(report_path)
        except Exception as e:
            print(f"⚠️ Failed to generate statsmodels summary report: {e}")

    results["audit_alerts"] = generate_audit_alerts(results, config)

    # Load baseline metrics for comparison (if configured)
    results['baseline_metrics'] = load_baseline_metrics(
        log_path=config.paths.metrics_log,
        baseline_id=config.evaluation.compare_to_baseline
    )

    return results