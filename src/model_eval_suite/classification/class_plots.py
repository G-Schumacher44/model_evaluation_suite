import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, Optional

"""
📊 Plotting suite for classification model evaluation.

Responsibilities:
- Generates and saves key evaluation visualizations including:
  • Confusion matrix
  • ROC and PR curves
  • Learning and calibration curves
  • Threshold and feature importance plots
  • Cumulative gains and lift curves
  • SHAP summary and bar plots (if enabled)
  • Feature coefficient plots (for linear models)
  • Distributions of numeric and categorical features
- Centralized orchestration via `generate_all_plots()`
- Uses SuiteConfig for consistent run ID tagging and export paths

This module is used during evaluation to create plots for both dashboards and reporting.
"""

import shap
from shap import Explanation
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from model_eval_suite.utils.config import SuiteConfig

def _get_run_save_dir(config: SuiteConfig, sub_dir: Optional[str] = None) -> Path:
    """Helper to create and return a plot save directory."""
    path = config.paths.plots_dir / config.run_id
    if sub_dir:
        path = path / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path

# --- HELPER FUNCTION TO HANDLE SHAP DIMENSIONALITY ---
def _prepare_shap_values_for_2d_plot(shap_explanation: Explanation) -> Explanation:
    """Ensures SHAP values are 2D for standard plots, slicing for the positive class if needed."""
    if shap_explanation.values.ndim == 3:
        # Slice SHAP values for the positive class if model is multiclass
        return shap_explanation[:, :, 1]
    return shap_explanation

# --- SHAP PLOT FUNCTIONS ---
def plot_shap_summary_beeswarm(shap_explanation: Explanation, config: SuiteConfig) -> Optional[str]:
    """Generates and saves a SHAP beeswarm summary plot."""
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"shap_summary_beeswarm_{config.run_id}.png"
    
    # Prepare SHAP values and generate beeswarm plot
    prepared_shap = _prepare_shap_values_for_2d_plot(shap_explanation)
    
    plt.figure()
    shap.plots.beeswarm(prepared_shap, show=False)
    plt.title(f"SHAP Summary (Beeswarm) - {config.run_id}")
    save_path = save_dir / f"shap_beeswarm_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return str(save_path)

def plot_shap_bar(shap_explanation: Explanation, config: SuiteConfig) -> Optional[str]:
    """Generates and saves a SHAP bar plot of mean absolute values."""
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"shap_bar_plot_{config.run_id}.png"
    
    # Prepare SHAP values and generate bar plot of mean absolute SHAP values
    prepared_shap = _prepare_shap_values_for_2d_plot(shap_explanation)
    
    plt.figure()
    shap.plots.bar(prepared_shap, show=False)
    plt.title(f"Mean Absolute SHAP Values - {config.run_id}")
    save_path = save_dir / f"shap_bar_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return str(save_path)

# --- STANDARD PLOTTING FUNCTIONS ---
def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, config: SuiteConfig) -> Dict[str, str]:
    dist_plot_paths = {}
    save_dir = _get_run_save_dir(config) / "distributions"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    factory_config = config.modeling.pipeline_factory
    numeric_features = factory_config.numeric_features
    categorical_features = factory_config.categorical_features
    
    df = X.copy()
    target_col_name = config.modeling.target_column
    df[target_col_name] = y
    
    # Plot histograms of numeric features grouped by target class
    for col in numeric_features:
        if col not in df.columns: continue
        fig, ax = plt.subplots(figsize=(8, 5))
        class_0_data = df[df[target_col_name] == 0][col].dropna()
        class_1_data = df[df[target_col_name] == 1][col].dropna()
        ax.hist([class_0_data, class_1_data], bins='auto', stacked=True, label=[f'Stay (Class 0)', f'Churn (Class 1)'], edgecolor='white')
        ax.set_title(f"Distribution of '{col}' by Target - {config.run_id}")
        ax.set_xlabel(col); ax.set_ylabel("Count"); ax.legend(); ax.grid(axis='y', alpha=0.5)
        save_path = save_dir / f"dist_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        dist_plot_paths[f"dist_{col}"] = str(save_path)
        
    # Plot count distributions of categorical features with hue by target class
    for col in categorical_features:
        if col not in df.columns: continue
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, hue=target_col_name)
        plt.title(f"Distribution of '{col}' by Target - {config.run_id}")
        plt.xticks(rotation=45, ha='right')
        save_path = save_dir / f"dist_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close()
        dist_plot_paths[f"dist_{col}"] = str(save_path)
        
    return dist_plot_paths

def plot_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"confusion_matrix_{config.run_id}.png"
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {config.run_id}")
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_roc_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"roc_curve_{config.run_id}.png"
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve - {config.run_id}")
    ax.plot([0, 1], [0, 1], 'k--', label='No Skill'); ax.legend()
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_pr_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"pr_curve_{config.run_id}.png"
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"Precision-Recall Curve - {config.run_id}")
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_learning_curve(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, config: SuiteConfig) -> Optional[str]:
    # Compute learning curve using 5-fold CV and plot mean scores across training sizes
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"learning_curve_{config.run_id}.png"
    train_sizes, train_scores, test_scores = learning_curve(estimator=model, X=X_train, y=y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1')
    train_scores_mean = np.mean(train_scores, axis=1); test_scores_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_title(f"Learning Curve - {config.run_id}"); ax.set_xlabel("Training examples"); ax.set_ylabel("F1 Score"); ax.legend(loc="best"); ax.grid(True)
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_calibration_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    if not hasattr(model, "predict_proba"): return None
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"calibration_curve_{config.run_id}.png"
    probs = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
    fig, ax = plt.subplots()
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives"); ax.set_title(f"Calibration Curve - {config.run_id}"); ax.legend(); ax.grid(True)
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_threshold_analysis(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    # Plot precision, recall, and F1 score across different classification thresholds
    if not hasattr(model, "predict_proba"): return None
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"threshold_analysis_{config.run_id}.png"
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
    fig, ax = plt.subplots()
    ax.plot(thresholds, precision[:-1], label="Precision"); ax.plot(thresholds, recall[:-1], label="Recall"); ax.plot(thresholds, f1_scores[:-1], label="F1-Score", lw=2)
    ax.set_title(f"Threshold Analysis - {config.run_id}"); ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score"); ax.legend(); ax.grid(True)
    plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    return str(save_path)

def plot_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    # Plot permutation feature importances with error bars for variance
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"permutation_importance_{config.run_id}.png"
    final_features = model.named_steps['preprocessor'].get_feature_names_out()
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1')

    importances_mean = result.importances_mean
    importances_std = result.importances_std
    sorted_idx = importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, max(5, len(final_features) * 0.5)))
    ax.barh(
        np.array(final_features)[sorted_idx],
        importances_mean[sorted_idx],
        xerr=importances_std[sorted_idx],
        align='center',
        error_kw=dict(lw=2, capsize=4, capthick=2, ecolor='#1f4e79')
    )
    ax.set_xlabel("Mean Decrease in F1 Score")
    ax.set_title(f"Permutation Importance - {config.run_id}")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return str(save_path)

def plot_cumulative_gain(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    # Plot cumulative gains curve based on sorted predicted probabilities
    if not hasattr(model, "predict_proba"): return None
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"cumulative_gain_{config.run_id}.png"
    y_probs = model.predict_proba(X_test)[:, 1]
    sorted_indices = np.argsort(y_probs)[::-1]; y_true_sorted = np.array(y_test)[sorted_indices]
    cumulative_positives = np.cumsum(y_true_sorted); total_positives = np.sum(y_true_sorted)
    percentages = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted); gains = cumulative_positives / total_positives
    fig, ax = plt.subplots()
    ax.plot(percentages, gains, label="Model"); ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("Percentage of sample"); ax.set_ylabel("Gain (Percentage of positives captured)"); ax.set_title(f"Cumulative Gains Curve - {config.run_id}"); ax.legend(); ax.grid(True)
    plt.savefig(save_path); plt.close(fig)
    return str(save_path)

def plot_lift_curve(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig) -> Optional[str]:
    # Plot lift curve comparing model performance to baseline at various thresholds
    if not hasattr(model, "predict_proba"): return None
    save_dir = _get_run_save_dir(config, "evaluation")
    save_path = save_dir / f"lift_curve_{config.run_id}.png"
    y_probs = model.predict_proba(X_test)[:, 1]
    sorted_indices = np.argsort(y_probs)[::-1]; y_true_sorted = np.array(y_test)[sorted_indices]
    cumulative_positives = np.cumsum(y_true_sorted); total_positives = np.sum(y_true_sorted)
    percentages = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    gains = cumulative_positives / total_positives; lift = gains / percentages
    fig, ax = plt.subplots()
    ax.plot(percentages, lift, label="Model"); ax.plot([0, 1], [1, 1], 'k--', label="Baseline")
    ax.set_xlabel("Percentage of sample"); ax.set_ylabel("Lift"); ax.set_title(f"Lift Curve - {config.run_id}"); ax.legend(); ax.grid(True)
    plt.savefig(save_path); plt.close(fig)
    return str(save_path)

def plot_feature_coefficients(model: Pipeline, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    """Generates a bar plot of feature coefficients for linear models."""
    # Bar plot of model coefficients from linear estimator (e.g., Logistic Regression)
    try:
        estimator = model.named_steps['estimator']
        if not hasattr(estimator, 'coef_'):
            return None 

        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        # Get the coefficients (for binary classification, coef_ is shape (1, n_features))
        coefs = estimator.coef_.flatten()

        coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs}).sort_values('coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.4)))
        sns.barplot(x='coefficient', y='feature', data=coef_df, ax=ax, orient='h')
        ax.set_title("Feature Coefficients (Log-Odds)")
        ax.grid(True)
        plt.tight_layout()
        
        save_path = save_dir / f"feature_coefficients_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    except Exception as e:
        print(f"⚠️ Could not generate coefficient plot: {e}")
        return None

# --- CV SCORE DISTRIBUTION PLOT ---
def plot_cv_score_distribution(results_data: Dict[str, Any], config: SuiteConfig, save_dir: Path) -> Optional[str]:
    """Creates a boxplot of the cross-validation scores."""
    cv_scores = results_data.get('metrics', {}).get('raw_cv_scores')
    if not cv_scores:
        return None  # Skip plot if no CV scores are found

    scoring_metric = config.modeling.hyperparameter_tuning.scoring

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=cv_scores, ax=ax, color='skyblue', width=0.3)
    sns.stripplot(data=cv_scores, ax=ax, color='black', size=5, jitter=0.1)

    ax.set_title(f'Cross-Validation Score Distribution ({len(cv_scores)} Folds)')
    ax.set_ylabel(f'{scoring_metric.title()} Score')
    ax.set_xticks([0])
    ax.set_xticklabels(['Scores per Fold'])
    ax.grid(axis='y')
    # Add small buffer around score range and overlay fold-level scores
    ax.set_ylim(min(cv_scores) - 0.01, max(cv_scores) + 0.01)  # Add a small buffer around score range

    save_path = save_dir / f"cv_score_distribution_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

# --- PLOT ORCHESTRATOR ---
def generate_all_plots(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig,
    results: Dict[str, Any]  # NEW ARGUMENT
) -> Dict[str, Optional[str]]:
    """Orchestrates the generation of all configured, non-SHAP plots."""

    save_dir = _get_run_save_dir(config, "evaluation")

    plot_paths = {}
    plot_configs = config.evaluation.plots

    # Map plot function names to their implementations
    plot_functions = {
        'confusion_matrix': plot_confusion_matrix,
        'roc_curve': plot_roc_curve,
        'pr_curve': plot_pr_curve,
        'learning_curve': plot_learning_curve,
        'calibration_curve': plot_calibration_curve,
        'threshold_plot': plot_threshold_analysis,
        'permutation_importance': plot_permutation_importance,
        'cumulative_gain': plot_cumulative_gain,
        'lift_curve': plot_lift_curve,
    }

    for name, func in plot_functions.items():
        if getattr(plot_configs, name).get('save', False):
            args = (model, X_train, y_train, config) if name == 'learning_curve' else (model, X_test, y_test, config)
            plot_paths[name] = func(*args)

    if plot_configs.feature_distributions.get('save', False):
        dist_paths = plot_feature_distributions(X_test, y_test, config)
        plot_paths.update(dist_paths)

    if isinstance(model.named_steps['estimator'], LogisticRegression):
        plot_paths['feature_coefficients'] = plot_feature_coefficients(model, config, save_dir)

    # Add CV score distribution plot if available
    if results.get("metrics", {}).get("raw_cv_scores"):
        plot_paths['cv_score_distribution'] = plot_cv_score_distribution(
            results_data=results, config=config, save_dir=save_dir
        )

    # Return dictionary of generated plot paths
    return {k: v for k, v in plot_paths.items() if v is not None}