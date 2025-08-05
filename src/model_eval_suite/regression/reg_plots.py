import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import statsmodels.api as sm
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from typing import Dict, Optional, Any

from model_eval_suite.utils.config import SuiteConfig

"""
📊 Regression Plotting Module

This module defines plotting utilities for evaluating regression models in the model evaluation suite.

Responsibilities:
- Generates predicted vs. actual plots, residual plots, Q-Q plots, and learning curves
- Computes and visualizes SHAP summaries and permutation importance
- Renders distribution plots for numeric and categorical features
- Supports modular plot generation controlled via the evaluation config YAML

Entrypoint:
- generate_all_plots(): orchestrates the creation and saving of all configured plots
"""

def _get_run_save_dir(config: SuiteConfig, sub_dir: Optional[str] = None) -> Path:
    # Create directory to save plots for current run
    path = config.paths.plots_dir / config.run_id
    if sub_dir:
        path = path / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path

def plot_predicted_vs_actual(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Scatter plot of true vs. predicted values with 45° reference line
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    x_lims, y_lims = ax.get_xlim(), ax.get_ylim()
    min_val, max_val = min(x_lims[0], y_lims[0]), max(x_lims[1], y_lims[1])
    lims = [min_val, max_val]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f"Predicted vs. Actual Values - {config.run_id}")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.grid(True)
    save_path = save_dir / f"predicted_vs_actual_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def plot_residuals(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Scatter plot of residuals (error) vs. predicted values
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title(f"Residuals Plot - {config.run_id}")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.grid(True)
    save_path = save_dir / f"residuals_plot_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def plot_residuals_histogram(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Histogram of residuals with KDE overlay
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f"Distribution of Residuals - {config.run_id}")
    ax.set_xlabel("Residual Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    save_path = save_dir / f"residuals_histogram_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def plot_qq(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Q-Q plot to assess normality of residuals
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title(f"Q-Q Plot of Residuals - {config.run_id}")
    save_path = save_dir / f"qq_plot_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def plot_learning_curve(model: Pipeline, X: pd.DataFrame, y: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Visualize training and validation scores as sample size increases
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model, X=X, y=y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5), scoring='r2'
        )
        train_scores_mean, test_scores_mean = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        ax.set_title(f"Learning Curve - {config.run_id}"); ax.set_xlabel("Training examples"); ax.set_ylabel("R² Score")
        ax.legend(loc="best"); ax.grid(True)
        save_path = save_dir / f"learning_curve_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        return str(save_path)
    except Exception as e:
        print(f"⚠️ Could not generate learning curve: {e}"); return None

def plot_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Visualize feature importance based on drop in R² when values are shuffled
    try:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
        importances_mean = result.importances_mean
        importances_std = result.importances_std
        sorted_idx = importances_mean.argsort()
        final_features = np.array(model.named_steps['preprocessor'].get_feature_names_out())[sorted_idx]

        fig, ax = plt.subplots(figsize=(10, max(6, len(final_features) * 0.4)))
        ax.barh(
            final_features,
            importances_mean[sorted_idx],
            xerr=importances_std[sorted_idx],
            align='center',
            error_kw=dict(lw=2, capsize=4, capthick=2, ecolor='#1f4e79')
        )
        ax.set_title(f"Permutation Importance - {config.run_id}")
        ax.set_xlabel("Mean Decrease in R² Score")
        fig.tight_layout()
        save_path = save_dir / f"permutation_importance_{config.run_id}.png"
        plt.savefig(save_path)
        plt.close(fig)
        return str(save_path)
    except Exception as e:
        print(f"⚠️ Could not generate permutation importance: {e}")
        return None

def plot_shap_summary(results_data: Dict[str, Any], config: SuiteConfig, plot_type: str, save_dir: Path) -> Optional[str]:
    # SHAP summary or bar plot depending on plot_type
    if 'shap_values' not in results_data: return None
    shap_values = results_data['shap_values']
    plt.figure()
    if plot_type == 'bar':
        shap.plots.bar(shap_values, show=False)
        plt.title(f"SHAP Feature Importance - {config.run_id}")
    else:
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Summary (Beeswarm) - {config.run_id}")
    save_path = save_dir / f"shap_summary_{plot_type}_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight"); plt.close()
    return str(save_path)

def plot_feature_coefficients(model: Pipeline, config: SuiteConfig, save_dir: Path) -> Optional[str]:
    # Bar plot of coefficients for linear models
    try:
        estimator = model.named_steps['estimator']
        if not hasattr(estimator, 'coef_'):
            return None
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        coefficients = estimator.coef_
        coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients}).sort_values('coefficient', ascending=False)
        fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.4)))
        sns.barplot(x='coefficient', y='feature', data=coef_df, ax=ax, orient='h')
        ax.set_title("Feature Coefficients")
        ax.grid(True)
        plt.tight_layout()
        save_path = save_dir / f"feature_coefficients_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return str(save_path)
    except Exception as e:
        print(f"⚠️ Could not generate coefficient plot: {e}")
        return None

def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, config: SuiteConfig) -> Dict[str, str]:
    dist_plot_paths = {}
    save_dir = _get_run_save_dir(config, "distributions")
    factory_config = config.modeling.pipeline_factory
    df = X.copy()
    target_name = y.name if y.name else 'target'
    df[target_name] = y

    # Scatter for numeric vs. target | Boxplot for categorical vs. target
    for col in factory_config.numeric_features:
        if col not in df.columns: continue
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=col, y=target_name, ax=ax, alpha=0.5)
        ax.set_title(f"'{col}' vs. Target Variable"); ax.grid(True)
        save_path = save_dir / f"dist_scatter_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        dist_plot_paths[f"dist_{col}"] = str(save_path)
        
    # Scatter for numeric vs. target | Boxplot for categorical vs. target
    for col in factory_config.categorical_features:
        if col not in df.columns: continue
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.boxplot(data=df, x=col, y=target_name, ax=ax)
        ax.set_title(f"Target Distribution by '{col}'"); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        save_path = save_dir / f"dist_box_{col}_{config.run_id}.png"
        plt.savefig(save_path, bbox_inches="tight"); plt.close(fig)
        dist_plot_paths[f"dist_{col}"] = str(save_path)
        
    return dist_plot_paths

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

    save_path = save_dir / f"cv_score_distribution_{config.run_id}.png"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)

def generate_all_plots(
    model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, 
    X_test: pd.DataFrame, y_test: pd.Series, config: SuiteConfig, results: Dict[str, Any]
) -> Dict[str, Optional[str]]:
    # Check config to conditionally generate each plot and collect paths
    
    eval_save_dir = _get_run_save_dir(config, "evaluation")
    plot_paths = {}
    plot_configs = config.evaluation.plots
    
    if plot_configs.predicted_vs_actual.get('save', False):
        plot_paths['predicted_vs_actual'] = plot_predicted_vs_actual(model, X_test, y_test, config, eval_save_dir)
    if plot_configs.residuals_plot.get('save', False):
        plot_paths['residuals_plot'] = plot_residuals(model, X_test, y_test, config, eval_save_dir)
    if plot_configs.residuals_histogram.get('save', False):
        plot_paths['residuals_histogram'] = plot_residuals_histogram(model, X_test, y_test, config, eval_save_dir)
    if plot_configs.qq_plot.get('save', False):
        plot_paths['qq_plot'] = plot_qq(model, X_test, y_test, config, eval_save_dir)
    if plot_configs.learning_curve.get('save', False):
        plot_paths['learning_curve'] = plot_learning_curve(model, X_train, y_train, config, eval_save_dir)
    if plot_configs.permutation_importance.get('save', False):
        plot_paths['permutation_importance'] = plot_permutation_importance(model, X_test, y_test, config, eval_save_dir)
        
    if 'shap_values' in results and plot_configs.shap_summary.get('save', False):
        plot_paths['shap_bar_plot'] = plot_shap_summary(results, config, 'bar', eval_save_dir)
        plot_paths['shap_beeswarm_plot'] = plot_shap_summary(results, config, 'beeswarm', eval_save_dir)

    if isinstance(model.named_steps['estimator'], LinearRegression):
        plot_paths['feature_coefficients'] = plot_feature_coefficients(model, config, eval_save_dir)
        
    if plot_configs.feature_distributions.get('save', False):
        dist_paths = plot_feature_distributions(X_test, y_test, config)
        plot_paths.update(dist_paths)

    # Add CV score distribution plot if available and valid
    cv_scores = results.get("metrics", {}).get("raw_cv_scores")
    if cv_scores and isinstance(cv_scores, (list, np.ndarray)) and all(np.isfinite(cv_scores)):
        plot_paths['cv_score_distribution'] = plot_cv_score_distribution(results, config, eval_save_dir)

    return {k: v for k, v in plot_paths.items() if v is not None}