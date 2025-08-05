import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
import shap
from IPython.display import display, HTML as IHTML
from pathlib import Path
from typing import Optional, Dict, Any

from model_eval_suite.utils.plot_viewer import PlotViewer
from model_eval_suite.utils.config import SuiteConfig

"""
🖥️ Classification Dashboard Utility

Interactive Jupyter dashboard for visualizing classification model evaluation results.

Components:
- Summary tab: Key test metrics and stability indicators
- Baseline comparison: Optional metrics comparison vs. baseline model
- Importance plots: Feature importance visualizations (coefficients, permutation, SHAP)
- Explainability: Local SHAP explanations with interactive controls
- Evaluation & Distribution plots: Rendered via PlotViewer widget
- Reports: Classification reports and statistical summaries
- Configuration: Displays full config JSON used for the run

Exposed functions:
- display_evaluation_dashboard(results): Main entrypoint to launch the dashboard.
"""

def _build_html_card(title: str, content: str, style: str = "") -> widgets.HTML:
    """Helper function to create styled card widgets with simple HTML content."""
    return widgets.HTML(f"""<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 8px 0; {style}">
                           <h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">{title}</h4>{content}</div>""")

class EvaluationDashboard:
    def __init__(self, results: Dict[str, Any]):
        self.config: SuiteConfig = results.get("config_obj")
        self.results = results
        self.metrics = results.get("metrics", {})
        self.baseline_metrics = results.get("baseline_metrics")
        self.plot_paths = results.get("plot_paths", {})
        self.widget = self._build_dashboard()

    def _render_banner(self) -> widgets.HTML:
        # Render a simple banner showing run ID and test F1 score
        f1 = self.metrics.get('test_f1', 0)
        run_id = self.config.run_id if self.config else "N/A"
        return widgets.HTML(f"""<div style="border: 1px solid #d0d7de; background-color: #f6f8fa; padding: 12px; border-radius: 6px; margin-bottom: 1em;">
                                <strong>Run:</strong> {run_id} | <strong>✅ F1 (Test):</strong> {f1:.3f}
                            </div>""")

    def _render_audit_alerts(self) -> widgets.HTML:
        # Show any warnings or audit alerts flagged during model evaluation
        alerts = self.results.get('audit_alerts', [])
        if not alerts:
            return widgets.HTML("")
        items = "".join([f"<li>{alert}</li>" for alert in alerts])
        return widgets.HTML(f"""<div style="border: 1px solid #f0b37e; background: #fff2e5; padding: 12px; border-radius: 6px; margin-top:1em;">
                                <strong>Audit Alerts:</strong><ul>{items}</ul>
                            </div>""")


    def _find_baseline_value(self, metric_keys: list) -> float | None:
        for key in metric_keys:
            if key in self.baseline_metrics:
                return self.baseline_metrics[key]
        return None

    def _build_summary_tab(self) -> widgets.VBox:
        # Create visual KPI cards for test performance metrics
        # and display train/test split and CV stability for overfitting diagnostics
        kpi_html_parts = []
        possible_metrics = [
            {"label": "F1 Score", "keys": ["test_f1"], "format": ".3f"},
            {"label": "Precision", "keys": ["test_precision"], "format": ".3f"},
            {"label": "Recall", "keys": ["test_recall"], "format": ".3f"},
            {"label": "ROC AUC", "keys": ["test_roc_auc"], "format": ".3f"}
        ]

        for metric_info in possible_metrics:
            found_key = next((key for key in metric_info['keys'] if key in self.metrics), None)
            if found_key:
                value = self.metrics[found_key]
                kpi_html = f'''<p style="font-size: 24px; font-weight: bold; margin: 0; color: #0366d6;">{value:{metric_info['format']}}</p>'''

                if self.baseline_metrics:
                    baseline_value = self._find_baseline_value(metric_info['keys'])
                    if baseline_value is not None:
                        delta_pct = ((value - baseline_value) / baseline_value) * 100 if baseline_value != 0 else 0
                        color = 'green' if delta_pct >= 0 else 'red'
                        sign = '+' if delta_pct >= 0 else ''
                        kpi_html += f'''<p style="font-size: 12px; margin: 0; color: {color};">{sign}{delta_pct:.2f}% vs. baseline</p>'''

                kpi_html_parts.append(
                    f'''<div style="text-align: center;">
                        <p style="font-size: 14px; margin: 0; color: #586069;">{metric_info['label']}</p>
                        {kpi_html}
                    </div>'''
                )

        kpi_html = f'''<div style="display: flex; justify-content: space-around;">{''.join(kpi_html_parts)}</div>'''

        train_f1 = self.metrics.get('train_f1', 0)
        test_f1 = self.metrics.get('test_f1', 0)
        cv_mean = self.metrics.get('cv_mean_f1', 0)
        cv_std = self.metrics.get('cv_std_f1', 0)
        stability_html = f"""<table style="width:100%; border-collapse: collapse;">
            <tr><td style="padding: 8px; border-bottom: 1px solid #e1e4e8;">Train vs. Test F1</td><td style="padding: 8px; border-bottom: 1px solid #e1e4e8; text-align: right; font-weight: bold;">{train_f1:.3f} vs. {test_f1:.3f}</td></tr>
            <tr><td style="padding: 8px;">Cross-Validation F1 (Mean ± Std)</td><td style="padding: 8px; text-align: right; font-weight: bold;">{cv_mean:.3f} ± {cv_std:.3f}</td></tr></table>"""

        # --- Add CV Score Distribution Plot ---
        cv_plot_path_str = self.plot_paths.get('cv_score_distribution')
        if cv_plot_path_str and Path(cv_plot_path_str).exists():
            import base64
            image_bytes = Path(cv_plot_path_str).read_bytes()
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            html_img = f'<img src="data:image/png;base64,{b64_image}" width="500"/>'
            cv_plot_card = _build_html_card("Cross-Validation Score Distribution", html_img)
        else:
            cv_plot_card = widgets.Box()  # Empty placeholder

        return widgets.VBox([
            _build_html_card("Test Set Performance", kpi_html),
            _build_html_card("Model Stability", stability_html),
            cv_plot_card  # Insert CV plot card here
        ])

    def _build_static_image_gallery(self, plot_paths: Dict[str, str]) -> widgets.VBox:
        # Display key feature importance plots (coefficients, permutation, SHAP) as static images
        # Only include plots that exist in the expected paths
        items = []
        importance_plot_keys = ['feature_coefficients', 'permutation_importance', 'shap_bar']
        for name, path_str in plot_paths.items():
            if not path_str:
                continue
            path = Path(path_str)
            if path.exists() and name in importance_plot_keys:
                image_bytes = path.read_bytes()
                image_widget = widgets.Image(value=image_bytes, format=path.suffix.lstrip('.'), width=600)

                display_name = name.replace('_', ' ').title()
                title_widget = widgets.HTML(f"<h4>{display_name}</h4>")

                items.append(widgets.VBox([title_widget, image_widget]))

        if not items:
            return widgets.VBox([widgets.HTML("<i>No importance plots were generated.</i>")])
        return widgets.VBox(items)

    def _build_plot_viewer_tab(self, title: str, sub_directory: Optional[str] = None) -> widgets.VBox:
        """Builds a tab content using the PlotViewer widget."""
        if not self.config: return widgets.VBox([widgets.HTML("<i>Configuration not available.</i>")])
        
        # Construct the directory path where evaluation plots are stored for this run
        plot_dir = self.config.paths.plots_dir / self.config.run_id
        if sub_directory:
            plot_dir = plot_dir / sub_directory
        # Ensure directory exists and contains at least one supported image before rendering
        
        if not plot_dir.is_dir() or not any(p.suffix.lower() in PlotViewer.SUPPORTED_FORMATS for p in plot_dir.iterdir()):
             return widgets.VBox([widgets.HTML(f"<p><em>No plot images found in '{sub_directory or 'main plots'}'.</em></p>")])
        
        viewer = PlotViewer(str(plot_dir), title=title)
        return widgets.VBox([viewer.widget_box])

    def _build_baseline_tab(self) -> widgets.HTML:
        # Compare current model’s test metrics to a baseline run, if provided
        # Highlight deltas for interpretability
        if not self.baseline_metrics: return widgets.HTML("<i>No baseline run specified.</i>")
        baseline_id = self.baseline_metrics.get('run_id', 'Baseline')
        metrics_to_compare = ['test_f1', 'test_precision', 'test_recall', 'test_roc_auc']
        data = [{'Metric': m.replace('test_', '').replace('_', ' ').title(),
                 'Current Model': f"{self.metrics.get(m, 0):.3f}",
                 f"Baseline ({baseline_id})": f"{self.baseline_metrics.get(m, 0):.3f}",
                 "Change": f"{self.metrics.get(m, 0) - self.baseline_metrics.get(m, 0):+.3f}"}
                for m in metrics_to_compare]
        df = pd.DataFrame(data)
        content = df.to_html(index=False, classes='table', border=0)
        return _build_html_card(f"Comparison to Baseline ({baseline_id})", content)

    def _build_explainer_tab(self) -> widgets.VBox:
        # Provide a local SHAP explainer for a selected row from the test sample
        # Button triggers dynamic force_plot rendering in notebook
        explainer = self.results.get('explainer')
        data_for_shap = self.results.get('data_for_shap')
        shap_values_obj = self.results.get('shap_values')
        
        if not all([explainer, data_for_shap is not None, shap_values_obj is not None]):
            return widgets.VBox([widgets.HTML("<i>SHAP analysis was not run or failed.</i>")])
        
        shap.initjs()
        description_widget = widgets.HTML("<p>Enter a row index from the SHAP sample to see its local explanation.</p>")
        index_input = widgets.IntText(value=0, description='Row Index:', style={'description_width': 'initial'})
        explain_button = widgets.Button(description="Explain Prediction")
        interactive_controls = widgets.HBox([index_input, explain_button])
        output_area = widgets.Output()
        
        def on_explain_click(b):
            with output_area:
                output_area.clear_output(wait=True)
                idx = index_input.value
                if not 0 <= idx < len(data_for_shap):
                    print(f"Error: Index must be between 0 and {len(data_for_shap) - 1}")
                    return

                try:
                    # Check if SHAP explainer and values are properly loaded
                    # Handle multiclass vs binary shap value slicing
                    if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
                        base_value = explainer.expected_value[1]
                        shap_values_for_instance = shap_values_obj.values[idx, :, 1]
                    else:
                        base_value = explainer.expected_value
                        shap_values_for_instance = shap_values_obj.values[idx]

                    force_plot = shap.force_plot(
                        base_value,
                        shap_values_for_instance,
                        data_for_shap.iloc[idx, :],
                        matplotlib=False
                    )
                    display(IHTML(force_plot.html()))
                except Exception as e:
                    print(f"⚠️ Failed to render SHAP force plot: {e}")

        explain_button.on_click(on_explain_click)
        
        title_widget = widgets.HTML('<h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">Local Prediction Explainer</h4>')
        content_widget = widgets.VBox([description_widget, interactive_controls, output_area])
        card_container = widgets.VBox([title_widget, content_widget], layout=widgets.Layout(
            border='1px solid #e1e4e8', 
            border_radius='6px', 
            padding='16px', 
            margin='8px 0'
        ))
        return card_container

    def _build_reports_tab(self) -> widgets.VBox:
        # Show train/test classification reports as preformatted text
        # Include statistical summary if generated by statsmodels
        report_widgets = []
        
        train_report = self.results.get('metrics', {}).get('train_classification_report', 'Not available.')
        test_report = self.results.get('metrics', {}).get('test_classification_report', 'Not available.')
        report_widgets.append(_build_html_card("Train Classification Report", f"<pre>{train_report}</pre>"))
        report_widgets.append(_build_html_card("Test Classification Report", f"<pre>{test_report}</pre>"))

        summary_path_str = self.plot_paths.get('statistical_summary')
        if summary_path_str:
            try:
                summary_path = Path(summary_path_str)
                if summary_path.exists():
                    summary_text = summary_path.read_text()
                    summary_card = _build_html_card("Statistical Summary (statsmodels)", f"<pre>{summary_text}</pre>")
                    report_widgets.append(summary_card)
            except Exception as e:
                print(f"⚠️ Could not render statistical summary report: {e}")
                
        return widgets.VBox(report_widgets)

    
    def _build_config_tab(self) -> widgets.HTML:
        # Display the full config object used in the run (as JSON)
        if not self.config:
            return widgets.HTML("<i>Configuration not available.</i>")
        
        config_str = self.config.model_dump_json(indent=2)
        return widgets.HTML(f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{config_str}</pre>")

    def _build_dashboard(self) -> widgets.VBox:
        # Assemble all dashboard tabs, add titles, and return the layout
        if not self.config: return widgets.VBox([widgets.HTML("Run failed: Configuration object not found.")])
        
        banner = self._render_banner()
        audit_alerts = self._render_audit_alerts()
        
        importance_plot_keys = ['feature_coefficients', 'permutation_importance', 'shap_bar']
        importance_plots = {k: v for k, v in self.plot_paths.items() if k in importance_plot_keys}
        
        tab_children = [
            self._build_summary_tab(),
            self._build_static_image_gallery(importance_plots),
            self._build_explainer_tab(),
            self._build_plot_viewer_tab(title="Evaluation Plots", sub_directory="evaluation"),
            self._build_plot_viewer_tab(title="Feature Distribution Plots", sub_directory="distributions"),
            self._build_reports_tab(),
            self._build_config_tab()
        ]
        
        tab_titles = [
            '📊 Summary', '🎯 Importance', '🧠 Explainability', '🖼️ Eval Plots',
            '📈 Distributions', '📝 Reports', '⚙️ Configuration'
        ]
        
        if self.baseline_metrics:
            tab_children.insert(1, self._build_baseline_tab())
            tab_titles.insert(1, '⚔️ Baseline')

        # Audit tab removed; audit alerts are now shown inline below the banner.
        
        main_tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            main_tabs.set_title(i, title)
            
        return widgets.VBox([banner, audit_alerts, main_tabs])

    def display(self):
        display(self.widget)

def display_evaluation_dashboard(results: Dict[str, Any]):
    """Instantiates and displays the evaluation dashboard in a notebook."""
    if "config_dict" in results and "config_obj" not in results:
        results["config_obj"] = results.get("config_dict") # Fallback for safety
    
    dashboard = EvaluationDashboard(results)
    dashboard.display()