import joblib
import pandas as pd
import csv
import base64
from pathlib import Path
from typing import Dict, Any

from model_eval_suite.utils.config import SuiteConfig

"""
📦 Export Utilities for Model Evaluation Suite

This module provides functions to export evaluation artifacts after a model run.

Responsibilities:
- Save trained model using joblib
- Generate and save an interactive HTML report with embedded plots and metrics
- Append evaluation results to a central CSV log for tracking
- Ensure output directories are created and managed per run_id

Entrypoints:
- export_artifacts(): Main export orchestrator for each run
- _generate_html_report(): Builds static HTML report from results
- _append_to_metrics_log(): Updates central metrics log file
"""

def _generate_html_report(results: Dict[str, Any], save_path: Path):
    """Generates a standalone HTML report with embedded plots."""
    try:
        # Extract run ID and evaluation outputs
        run_id = results.get("config_obj").run_id if results.get("config_obj") else "N/A"
        metrics = results.get('metrics', {})
        plot_paths = results.get('plot_paths', {})

        # Begin HTML structure and add basic styles
        html = f"<html><head><title>Evaluation Report: {run_id}</title>"
        html += """<style>
            body { font-family: sans-serif; margin: 2em; color: #333; }
            h1, h2, h3 { color: #111; border-bottom: 1px solid #ddd; padding-bottom: 0.25em; }
            table { border-collapse: collapse; width: 80%; margin-bottom: 2em; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .plot-container { margin-bottom: 2em; padding: 1em; border: 1px solid #eee; border-radius: 5px; page-break-inside: avoid; }
            img { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; }
            pre { background-color: #f6f8fa; padding: 1em; border-radius: 5px; white-space: pre-wrap; word-wrap: break-wrap; }
        </style></head><body>"""
        html += f"<h1>Evaluation Report: {run_id}</h1>"
        
        # Filter and format performance metrics table
        metrics_filtered = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str)) and 'report' not in k}
        metrics_df = pd.DataFrame([metrics_filtered])
        html += "<h2>Performance Metrics</h2>"
        html += metrics_df.to_html(index=False, classes='table')
        
        # Add Text-Based Reports
        html += "<h2>Detail Reports</h2>"
        
        # Insert classification reports if available
        train_class_report = metrics.get('train_classification_report', '')
        if train_class_report:
            html += "<h3>Train Classification Report</h3>"
            html += f"<pre>{train_class_report}</pre>"
            
        test_class_report = metrics.get('test_classification_report', '')
        if test_class_report:
            html += "<h3>Test Classification Report</h3>"
            html += f"<pre>{test_class_report}</pre>"

        # Embed statistical summary from text report
        summary_path_str = plot_paths.get('statistical_summary')
        if summary_path_str:
            summary_path = Path(summary_path_str)
            if summary_path.exists():
                summary_text = summary_path.read_text()
                html += "<h3>Statistical Summary (statsmodels)</h3>"
                html += f"<pre>{summary_text}</pre>"
        
        # Loop through saved plot images and embed in HTML
        html += "<h2>Plots</h2>"
        # --- CORRECTED: Iterate through all plot paths ---
        for name, path_str in plot_paths.items():
            if not path_str or name == 'statistical_summary': continue # Skip the text report
            path = Path(path_str)
            if path.exists():
                with open(path, 'rb') as f:
                    encoded_img = base64.b64encode(f.read()).decode('utf-8')
                
                plot_title = name.replace('_', ' ').title()
                html += f"<div class='plot-container'><h3>{plot_title}</h3>"
                html += f"<img src='data:image/png;base64,{encoded_img}'>"
                html += "</div>"
        
        html += "</body></html>"
        
        # Write the final HTML report to disk
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
            
    except Exception as e:
        print(f"⚠️ Failed to generate HTML report: {e}")


def export_artifacts(results: Dict[str, Any], model: Any, config: SuiteConfig):
    """Saves model, results, and other artifacts to the specified directories."""
    run_id = config.run_id
    paths = config.paths
    
    # Create output directories for the current run
    model_export_path = paths.model_export_dir / run_id
    report_export_path = paths.reports_dir / run_id
    model_export_path.mkdir(parents=True, exist_ok=True)
    report_export_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model using joblib
    model_filepath = model_export_path / f"model_{run_id}.pkl"
    joblib.dump(model, model_filepath)
    
    # Append summary metrics to central log
    _append_to_metrics_log(results, run_id, paths.metrics_log)

    # Generate static HTML report if enabled
    if config.evaluation.export_html_dashboard:
        html_filepath = report_export_path / f"static_report_{run_id}.html"
        _generate_html_report(results, html_filepath)

def _append_to_metrics_log(results: Dict[str, Any], run_id: str, log_path: Path):
    """Appends key metrics for the current run to a central CSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = results.get('metrics', {})
    config_obj: SuiteConfig = results.get("config_obj")
    model_name = config_obj.modeling.pipeline_factory.name if config_obj else "Unknown"
    task_type = config_obj.task_type if config_obj else "unknown"

    is_regression = "r_squared" in metrics
    timestamp = pd.Timestamp.now().isoformat()

    if is_regression:
        log_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'task_type': task_type,
            'r_squared': round(metrics.get("r_squared", 0), 5),
            'root_mean_squared_error': round(metrics.get("root_mean_squared_error", 0), 5),
            'mean_absolute_error': round(metrics.get("mean_absolute_error", 0), 5)
        }
    else:
        log_data = {
            'run_id': run_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'task_type': task_type,
            'test_roc_auc': round(metrics.get("test_roc_auc", 0), 5),
            'cv_mean_f1': round(metrics.get("cv_mean_f1", 0), 5),
            'cv_std_f1': round(metrics.get("cv_std_f1", 0), 5),
            'train_f1': round(metrics.get("train_f1", 0), 5),
            'test_f1': round(metrics.get("test_f1", 0), 5),
            'test_precision': round(metrics.get("test_precision", 0), 5),
            'test_recall': round(metrics.get("test_recall", 0), 5)
        }

    # Reload full log if it exists and harmonize schema
    all_logs = []
    file_exists = log_path.exists()
    if file_exists:
        try:
            existing_df = pd.read_csv(log_path)
            all_logs = existing_df.to_dict(orient='records')
        except Exception as e:
            print(f"⚠️ Failed to read existing log for merge: {e}")

    all_logs.append(log_data)

    # Rebuild unified schema
    all_keys = [
        'run_id', 'timestamp', 'model_name', 'task_type',
        'test_roc_auc', 'cv_mean_f1', 'cv_std_f1', 'train_f1', 'test_f1', 'test_precision', 'test_recall',
        'r_squared', 'root_mean_squared_error', 'mean_absolute_error'
    ]
    all_logs_padded = [{k: row.get(k, "") for k in all_keys} for row in all_logs]

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(all_logs_padded)