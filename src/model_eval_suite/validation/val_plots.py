# src/model_eval_suite/validation/plots.py
# src/model_eval_suite/validation/plots.py
"""
📊 Validation Plots Module

This module generates diagnostic and interpretability plots for the final
evaluation of a trained model on the holdout dataset. It supports both
classification and regression tasks.

Key Features:
- Feature distribution plots by class or category
- Accuracy confidence intervals using Wilson score method
- F1-score and residual summaries by segment
- Predicted vs. Actual with 95% prediction intervals for ensemble regressors

Entrypoints:
- plot_feature_distributions_classification
- plot_accuracy_confidence_interval
- plot_performance_by_segment
- plot_residuals_by_segment
- plot_predicted_vs_actual_with_intervals
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
from statsmodels.stats.proportion import proportion_confint
from ..utils.config import SuiteConfig
from sklearn.metrics import f1_score

def _get_run_save_dir(config: SuiteConfig, sub_dir: str) -> Path:
    """Helper to create and return a plot save directory."""
    path = config.paths.plots_dir / config.run_id / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_feature_distributions_classification(X: pd.DataFrame, y: pd.Series, config: SuiteConfig) -> Dict[str, str]:
    """
    Generates plots showing feature distributions for each class in the target variable.
    """
    dist_plot_paths = {}
    save_dir = _get_run_save_dir(config, "distributions")
    factory_config = config.modeling.pipeline_factory
    # Combine features and target
    df = X.copy()
    target_name = y.name if y.name else 'target'
    df[target_name] = y

    # Plot distributions for numeric features
    for col in factory_config.numeric_features:
        if col not in df.columns: continue
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue=target_name, kde=True, ax=ax, palette="viridis")
        ax.set_title(f"Distribution of '{col}' by {target_name}")
        save_path = save_dir / f"dist_hist_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        dist_plot_paths[f"dist_hist_{col}"] = str(save_path)

    # Plot distributions for categorical features
    for col in factory_config.categorical_features:
        if col not in df.columns: continue
        ct = pd.crosstab(df[col], df[target_name], normalize='index')
        fig, ax = plt.subplots(figsize=(12, 7))
        ct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title(f"Proportion of {target_name} by '{col}'")
        ax.set_ylabel("Proportion")
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        save_path = save_dir / f"dist_bar_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        dist_plot_paths[f"dist_bar_{col}"] = str(save_path)
        
    return dist_plot_paths


def plot_accuracy_confidence_interval(y_true: pd.Series, y_pred: pd.Series, config: SuiteConfig) -> str:
    """Calculates and plots the 95% confidence interval for accuracy."""
    save_dir = _get_run_save_dir(config, "evaluation")
    
    # Calculate accuracy and CI
    accuracy = (y_true == y_pred).mean()
    n_samples = len(y_true)
    n_correct = (y_true == y_pred).sum()

    ci_low, ci_upp = proportion_confint(n_correct, n_samples, method='wilson')
    
    # Plot confidence interval with error bar
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.errorbar(x=accuracy, y=0, xerr=[[accuracy - ci_low], [ci_upp - accuracy]],
                capsize=5, capthick=2, ecolor='blue', marker='s', mfc='red', mec='red')
    
    ax.set_yticks([])
    ax.set_xlim(0.8, 1.0)
    ax.set_title('95% Confidence Interval for Accuracy')
    ax.set_xlabel('Accuracy')
    ax.grid(axis='x')
    
    save_path = save_dir / f"accuracy_ci_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def plot_performance_by_segment(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, segment_column: str, config: SuiteConfig) -> str:
    """Calculates and plots the F1 score for each category in a segment column."""
    save_dir = _get_run_save_dir(config, "evaluation")
    
    data = X.copy()
    data['true'] = y_true
    data['pred'] = y_pred
    
    # Compute F1 per segment
    f1_scores = data.groupby(segment_column).apply(lambda g: f1_score(g['true'], g['pred']))
    
    # Plot segment-wise F1 scores
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_scores.sort_values().plot(kind='barh', ax=ax, color='skyblue')
    ax.set_title(f'F1-Score by {segment_column}')
    ax.set_xlabel('F1-Score')
    ax.set_ylabel(segment_column)
    ax.grid(axis='x')
    
    save_path = save_dir / f"segment_perf_{segment_column}_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_residuals_by_segment(X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, segment_column: str, config: SuiteConfig) -> str:
    """Creates a boxplot of residuals for each category in a segment column."""
    save_dir = _get_run_save_dir(config, "evaluation")
    
    data = X.copy()
    # Calculate residuals
    data['residuals'] = y_true - y_pred
    
    # Plot residuals by segment category
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(data=data, x=segment_column, y='residuals', ax=ax, 
                hue=segment_column, palette="viridis", legend=False)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f"Model Residuals by {segment_column}")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = save_dir / f"segment_residuals_{segment_column}_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

# In src/model_eval_suite/validation/plots.py
import numpy as np

def plot_predicted_vs_actual_with_intervals(model: 'Pipeline', X: pd.DataFrame, y: pd.Series, config: SuiteConfig) -> Optional[str]:
    """
    Creates a Predicted vs. Actual plot with 95% prediction intervals.
    This implementation is for Random Forest Regressor models.
    """
    save_dir = _get_run_save_dir(config, "evaluation")
    estimator = model.named_steps['estimator']

    # Check for ensemble support
    if not hasattr(estimator, 'estimators_'):
        print("⚠️ Prediction interval plot is only available for ensemble models like RandomForestRegressor.")
        return None

    # Collect tree-level predictions
    tree_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
    
    # Compute 95% prediction intervals
    lower_bound = np.percentile(tree_preds, 2.5, axis=0)
    upper_bound = np.percentile(tree_preds, 97.5, axis=0)
    
    # Get the main prediction
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y, y_pred, alpha=0.5, label='Prediction')
    # Draw interval lines for each prediction
    for i in range(len(y)):
        ax.plot([y.iloc[i], y.iloc[i]], [lower_bound[i], upper_bound[i]], 'r-', alpha=0.2)

    # Identity line setup
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Perfect Prediction')
    
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('Predicted vs. Actual with 95% Prediction Intervals')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.grid(True)

    save_path = save_dir / f"pred_vs_actual_intervals_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)