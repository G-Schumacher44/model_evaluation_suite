"""
🔬 Pre-Model Diagnostics Module

This utility performs automated data quality and statistical checks before model training.

Responsibilities:
- Detect missing values, skewness, and outliers in numeric features
- Calculate and visualize VIF (Variance Inflation Factor) and correlation matrices
- Save plots and CSV reports to diagnostics subdirectories
- Optionally generate a standalone HTML diagnostics report

Entrypoint:
- run_pre_model_diagnostics(): orchestrates all checks and exports based on config
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from typing import Dict, Any
import base64


# INDIVIDUAL CHECK FUNCTIONS
def run_outlier_check(df: pd.DataFrame, target_column: str = None, save_path: Path = None) -> pd.DataFrame:
    """
    Identifies outliers using the IQR method and generates box plots.
    """
    # Select numeric columns, optionally excluding target
    numeric_df = df.select_dtypes(include=np.number)
    if target_column and target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])

    outlier_summary = {}
    # Calculate IQR and identify outliers for each column
    for col in numeric_df.columns:
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
        if not outliers.empty:
            outlier_summary[col] = len(outliers)

    import math

    # Generate and save boxplots for numeric features
    if save_path and not numeric_df.empty:
        n_cols = 3
        n_rows = math.ceil(len(numeric_df.columns) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_df.columns):
            sns.boxplot(data=numeric_df, y=col, ax=axes[i])
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    return pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['outlier_count'])

def check_skewness(df: pd.DataFrame, target_column: str = None, skew_threshold: float = 0.75) -> pd.DataFrame:
    """Calculates skewness for numeric features."""
    # Select numeric columns, optionally excluding target
    numeric_df = df.select_dtypes(include=np.number)
    if target_column and target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])
        
    # Calculate absolute skewness and filter by threshold
    skew_data = numeric_df.skew().abs()
    skewed_features = skew_data[skew_data > skew_threshold]
    
    return pd.DataFrame(skewed_features, columns=['skewness'])

def run_vif_check(df: pd.DataFrame, target_column: str = None, vif_threshold: float = 5.0, save_path: Path = None) -> pd.DataFrame:
    """
    Calculates and optionally saves a plot of the Variance Inflation Factor (VIF).
    """
    # Select numeric columns and remove near-constant ones
    numeric_df = df.select_dtypes(include=np.number)
    
    if target_column and target_column in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_column])
        
    numeric_df = numeric_df.loc[:, numeric_df.var() > 1e-6].dropna()

    if numeric_df.empty:
        return pd.DataFrame()

    # Calculate VIF for remaining features
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

    # Plot and save VIF bar chart
    if save_path:
        fig, ax = plt.subplots(figsize=(10, max(6, len(vif_data) * 0.3)))
        sns.barplot(data=vif_data, x="VIF", y="feature", hue="feature", palette="viridis", ax=ax, dodge=False)
        
        # --- CORRECTED: Check if legend exists before removing ---
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        ax.axvline(x=vif_threshold, color="red", linestyle="--", label=f"Threshold = {vif_threshold}")
        ax.set_title("Variance Inflation Factor (VIF)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    return vif_data

def run_correlation_check(df: pd.DataFrame, target_column: str = None, method: str = "pearson", save_path: Path = None) -> pd.DataFrame:
    """
    Calculates and optionally saves a heatmap of the correlation matrix.
    
    This function automatically selects numeric features and excludes the target column.
    """
    # Select numeric columns, optionally retaining target
    numeric_df = df.select_dtypes(include=np.number)
    
    if target_column and target_column in numeric_df.columns:
        # Keep target for correlation analysis with other features, but can be configured
        pass

    # Compute correlation matrix using specified method
    corr_matrix = numeric_df.corr(method=method)

    # Save annotated heatmap of correlations
    if save_path:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
        # Rotate x-axis tick labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(f"{method.title()} Correlation Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    return corr_matrix

def check_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary of missing values by column."""
    # Summarize missing counts and percentages by column
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    summary = pd.DataFrame({"missing_count": missing, "missing_percent": percent})
    return summary[summary["missing_count"] > 0].sort_values("missing_percent", ascending=False)

# ==============================================================================
# HTML REPORT GENERATION (NEW)
# ==============================================================================

def _generate_diagnostic_html_report(
    results: Dict[str, pd.DataFrame], 
    plot_paths: Dict[str, Path],
    save_path: Path,
    run_id: str
):
    """Generates a standalone HTML report for the diagnostics."""
    try:
        # Construct HTML structure and embed styles
        html = f"<html><head><title>Pre-Model Diagnostic Report: {run_id}</title>"
        html += """<style>
            body { font-family: sans-serif; margin: 2em; color: #333; }
            h1, h2 { color: #111; border-bottom: 1px solid #ddd; padding-bottom: 0.25em; }
            table { border-collapse: collapse; width: 80%; margin-bottom: 2em; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot-container { margin-bottom: 2em; padding: 1em; border: 1px solid #eee; border-radius: 5px; page-break-inside: avoid; }
            img { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }
        </style></head><body>"""
        html += f"<h1>Pre-Model Diagnostic Report: {run_id}</h1>"

        # Insert diagnostic tables into HTML
        for name, df in results.items():
            if not df.empty:
                title = name.replace('_', ' ').title()
                html += f"<h2>{title}</h2>"
                html += df.to_html(classes='table')

        # Embed diagnostic plots as base64 images
        html += "<h2>Diagnostic Plots</h2>"
        for name, path in plot_paths.items():
            if path.exists():
                with open(path, 'rb') as f:
                    encoded_img = base64.b64encode(f.read()).decode('utf-8')
                
                plot_title = name.replace('_', ' ').title()
                html += f"<div class='plot-container'><h2>{plot_title}</h2>"
                html += f"<img src='data:image/png;base64,{encoded_img}'>"
                html += "</div>"
                
        html += "</body></html>"
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
    except Exception as e:
        print(f"⚠️ Failed to generate diagnostic HTML report: {e}")


def run_pre_model_diagnostics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates a unified suite of pre-model diagnostics based on a config object.
    """
    # Skip diagnostics if not enabled in config
    diag_config = config.get("pre_model_diagnostics", {})
    if not diag_config.get("run", False):
        return {}
        
    # Retrieve identifiers and diagnostic output paths
    run_id = config.get("run_id", "default_run")
    target_column = config.get("modeling", {}).get("target_column")
    
    # --- CORRECTED: Define dedicated subdirectories for diagnostic artifacts ---
    base_reports_dir = Path(config['paths']['reports_dir']) / run_id
    base_plots_dir = Path(config['paths']['plots_dir']) / run_id
    
    diagnostic_reports_dir = base_reports_dir / "diagnostics"
    diagnostic_plots_dir = base_plots_dir / "diagnostics"
    
    # Ensure subdirectories for reports and plots exist
    diagnostic_reports_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths for each diagnostic plot
    plot_paths = {
        "vif_plot": diagnostic_plots_dir / "vif_plot.png",
        "correlation_matrix": diagnostic_plots_dir / "correlation_matrix.png",
        "outlier_boxplots": diagnostic_plots_dir / "outlier_boxplots.png"
    }

    # Run all individual diagnostic checks
    diagnostics_results = {
        "missingness_summary": check_missingness(df),
        "skewness_summary": check_skewness(df, target_column, diag_config.get("skewness_threshold", 0.75)),
        "outlier_summary": run_outlier_check(df, target_column, plot_paths["outlier_boxplots"]),
        "vif_scores": run_vif_check(df, target_column, diag_config.get("vif_threshold", 5.0), plot_paths["vif_plot"]),
        "correlation_matrix": run_correlation_check(df, target_column, save_path=plot_paths["correlation_matrix"])
    }

    # Export each diagnostic summary to CSV
    if diag_config.get("export_reports", True):
        for name, result_df in diagnostics_results.items():
            if not result_df.empty:
                result_df.to_csv(diagnostic_reports_dir / f"{name}.csv")

    # Optionally generate a consolidated HTML report
    if diag_config.get("export_html_report", True):
        html_save_path = diagnostic_reports_dir / "diagnostic_report.html"
        _generate_diagnostic_html_report(diagnostics_results, plot_paths, html_save_path, run_id)

    return diagnostics_results