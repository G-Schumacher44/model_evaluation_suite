from typing import Any, Dict, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from .class_metrics import generate_audit_alerts
from ..modeling import explainers
from . import class_plots
from model_eval_suite.utils.config import SuiteConfig
from . import class_metrics

"""
🎼 Evaluation of Classifiers Models

Orchestrates the end-to-end evaluation process for classification models.

Responsibilities:
- Computes classification metrics for training and test sets
- Handles probability predictions safely (with fallback if not available)
- Loads optional baseline metrics for comparison
- Generates SHAP and permutation explainers if enabled
- Saves visualizations (e.g., ROC, PR, SHAP) via the plotting suite
- For LogisticRegression models, optionally generates a statistical summary using statsmodels
- Aggregates results and audit alerts for dashboard consumption

Main Function:
- orchestrate_model_evaluation(): Takes a trained pipeline and config, returns a dictionary of evaluation results.
"""

def orchestrate_model_evaluation(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig,
    results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    
    # Require results dictionary to be provided by the caller
    if results is None:
        raise ValueError("Results dictionary must be provided by the caller.")
    results["config_obj"] = config
    results["config_dict"] = config.model_dump()

    # Compute train/test classification metrics, preserving any existing metrics like raw_cv_scores
    results.setdefault('metrics', {}).update(
        class_metrics.generate_model_metrics(model, X_train, y_train, X_test, y_test)
    )

    # Predict labels and probabilities (if available)
    results['y_pred'] = model.predict(X_test)
    try:
        results['y_pred_proba'] = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        results['y_pred_proba'] = None

    # Load baseline metrics for comparison (if configured)
    results['baseline_metrics'] = class_metrics.load_baseline_metrics(
        log_path=config.paths.metrics_log,
        baseline_id=config.evaluation.compare_to_baseline
    )

    # Generate SHAP and permutation explainers (if enabled)
    explainer_data = explainers.generate_all_explainers(model, X_train, config)
    results.update(explainer_data)

    # Generate evaluation plots (ROC, PR, etc.)
    plot_paths = class_plots.generate_all_plots(
        model, X_train, y_train, X_test, y_test, config, results
    )
    
    # Generate SHAP summary plots if enabled in config
    if 'shap_values' in results and config.evaluation.plots.shap_summary.get('save', False):
        plot_paths['shap_summary_beeswarm'] = class_plots.plot_shap_summary_beeswarm(
            results['shap_values'], config
        )
        plot_paths['shap_bar'] = class_plots.plot_shap_bar(
            results['shap_values'], config
        )

    # If estimator is LogisticRegression, generate a statistical summary using statsmodels
    if isinstance(model.named_steps['estimator'], LogisticRegression):
        try:
            import statsmodels.api as sm

            transformer_pipeline = model[:-1]
            X_train_transformed = transformer_pipeline.transform(X_train)
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()

            # Convert transformed training data to DataFrame with original index and feature names
            X_train_transformed_df = pd.DataFrame(
                X_train_transformed, 
                columns=feature_names, 
                index=X_train.index
            )

            # Add intercept term for logistic regression
            X_train_with_const = sm.add_constant(X_train_transformed_df, has_constant='add')

            # Fit logistic regression model using statsmodels for coefficient summary
            logit_model = sm.Logit(y_train, X_train_with_const).fit(disp=0)
            
            # Save summary report to disk and add path to outputs
            report_path = config.paths.reports_dir / config.run_id / "statistical_summary.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(logit_model.summary().as_text())
            
            plot_paths['statistical_summary'] = str(report_path)

        except ImportError:
            print("⚠️ statsmodels is not installed or has a version conflict. Cannot generate statistical summary.")
        except Exception as e:
            print(f"⚠️ Failed to generate statsmodels summary report: {e}")

    results['plot_paths'] = plot_paths


    # Run post-hoc audit checks on the evaluation results
    results['audit_alerts'] = class_metrics.generate_audit_alerts(results, config)

    # Return complete evaluation output dictionary
    return results
