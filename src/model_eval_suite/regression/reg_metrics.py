import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional
from model_eval_suite.utils.config import SuiteConfig
from pathlib import Path

"""
📊 Regression Metrics Module

This module defines utility functions to compute common regression performance metrics
from a fitted scikit-learn pipeline. Metrics include:

- R-squared (coefficient of determination)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

Used by the regression evaluator to generate consistent performance summaries.
"""

def generate_regression_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Calculates a standard set of metrics for a fitted regression model.
    """
    
    y_pred = model.predict(X_test)  # Predict on test set

    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared (coefficient of determination)

    metrics = {
        'r_squared': r2,  # Goodness of fit
        'mean_absolute_error': mae,  # Average magnitude of errors
        'mean_squared_error': mse,  # Average squared error
        'root_mean_squared_error': rmse,  # RMSE for interpretability
    }
    
    return metrics

def load_baseline_metrics(log_path: str, baseline_id: Optional[str]) -> Optional[Dict]:
    """
    Loads baseline metrics for a given run_id from the metrics log CSV.
    """
    if not baseline_id:
        return None

    log_file = Path(log_path)
    if not log_file.exists():
        return None

    try:
        df_log = pd.read_csv(log_path)
        baseline_rows = df_log[df_log["run_id"] == baseline_id]
        if baseline_rows.empty:
            print(f"⚠️ No baseline found for comparison ID: {baseline_id}")
            return None
        result = baseline_rows.iloc[-1].to_dict()
        result['run_id'] = baseline_id  
        return result
    except Exception as e:
        print(f"⚠️ Failed to load or parse baseline comparison metrics: {e}")
        return None
    
def generate_audit_alerts(results: Dict, config: SuiteConfig) -> list:
    """Generates a list of regression-focused warnings."""
    alerts = []
    current = results.get('metrics', {})
    baseline = results.get('baseline_metrics')
    audit_config = config.evaluation.audits

    # Audit 1: R² too low
    r2 = current.get('r_squared', 0)
    if r2 < audit_config.r_squared_min_threshold:
        alerts.append(f"⚠️ Weak Fit: R-squared is low ({r2:.3f}), indicating poor model fit.")

    # Audit 2: High RMSE
    rmse = current.get('root_mean_squared_error', 0)
    if rmse > audit_config.rmse_max_threshold:
        alerts.append(f"⚠️ High Error: RMSE is high ({rmse:,.2f}), indicating large prediction error.")

    # Audit 3: Regression vs baseline
    if baseline:
        baseline_r2 = baseline.get('r_squared', r2)
        if r2 < baseline_r2 * audit_config.performance_regression_r2_threshold_factor:
            alerts.append(f"⚠️ Regression Performance Drop: R² ({r2:.3f}) is more than 5% lower than baseline ({baseline_r2:.3f}).")

    return alerts