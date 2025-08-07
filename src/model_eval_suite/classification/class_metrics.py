import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from model_eval_suite.utils.config import SuiteConfig

"""
📊 Metrics module for evaluating classification models.

Includes:
- generate_model_metrics: Calculates F1, precision, recall, ROC AUC, CV scores, and classification reports.
- load_baseline_metrics: Loads prior run metrics from a CSV log for comparison.
- generate_audit_alerts: Produces warnings for overfitting, CV instability, and performance regressions.

Used during evaluation to generate performance summaries and diagnostics.
"""

def generate_model_metrics(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Calculates a comprehensive set of metrics for a fitted model."""
    
    # Predict labels for train and test sets
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    metrics = {}
    
    # Compute ROC AUC if model supports probability predictions
    try:
        y_proba_test = model.predict_proba(X_test)[:, 1]
        metrics['test_roc_auc'] = roc_auc_score(y_test, y_proba_test)
    except (AttributeError, IndexError):
        metrics['test_roc_auc'] = None

    # Evaluate model using 5-fold cross-validation on training data
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        metrics['cv_scores'] = cv_scores.tolist()
        metrics['cv_mean_f1'] = np.mean(cv_scores)
        metrics['cv_std_f1'] = np.std(cv_scores)
    except Exception:
        metrics['cv_scores'] = []
        metrics['cv_mean_f1'] = 0.0
        metrics['cv_std_f1'] = 0.0

    # Compute core classification metrics (F1, precision, recall) and full reports
    metrics.update({
        'train_f1': f1_score(y_train, y_pred_train),
        'test_f1': f1_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'train_classification_report': classification_report(y_train, y_pred_train),
        'test_classification_report': classification_report(y_test, y_pred_test),
    })
    
    return metrics

def load_baseline_metrics(log_path: str, baseline_id: Optional[str]) -> Optional[Dict]:
    """
    Loads baseline metrics for a given run_id from the metrics log CSV.
    """
    # Skip if no baseline comparison was requested
    if not baseline_id:
        return None
    
    # Verify that the metrics log file exists
    log_file = Path(log_path)
    if not log_file.exists():
        return None

    try:
        df_log = pd.read_csv(log_path)
        # Filter log for matching run_id
        baseline_rows = df_log[df_log["run_id"] == baseline_id]
        
        if baseline_rows.empty:
            print(f"⚠️ No baseline found for comparison ID: {baseline_id}")
            return None
            
        # Return the most recent matching baseline entry
        return baseline_rows.iloc[-1].to_dict()
    except Exception as e:
        print(f"⚠️ Failed to load or parse baseline comparison metrics: {e}")
        return None

def generate_audit_alerts(results: Dict, config: SuiteConfig) -> list:
    """Generates a list of warnings based on model performance metrics."""
    alerts = []
    current = results.get('metrics', {})
    baseline = results.get('baseline_metrics')  # Can be None
    audit_config = config.evaluation.audits

    # Check for overfitting based on F1 score gap
    train_f1 = current.get('train_f1', 0)
    test_f1 = current.get('test_f1', 0)
    if train_f1 > test_f1 * audit_config.overfitting_threshold_factor:
        alerts.append(f"⚠️ Overfitting Signal: Training F1 ({train_f1:.3f}) is significantly higher than Test F1 ({test_f1:.3f}).")

    # Check for high variance across CV folds
    cv_std = current.get('cv_std_f1', 0)
    if cv_std > audit_config.cv_std_threshold:
        alerts.append(f"⚠️ High CV Variance: F1 standard deviation across folds is high ({cv_std:.3f}).")

    # Compare current test F1 to baseline test F1
    if baseline:
        baseline_f1 = baseline.get('test_f1', 1.0)
        if test_f1 < baseline_f1 * audit_config.performance_regression_f1_threshold_factor:
            alerts.append(f"⚠️ Performance Regression: Test F1 ({test_f1:.3f}) is >5% lower than baseline ({baseline_f1:.3f}).")

    return alerts