import ipywidgets as widgets
import pandas as pd
import shap
from IPython.display import display, HTML as IHTML
from pathlib import Path
from typing import Dict, Any, Optional

from model_eval_suite.utils.config import SuiteConfig
from model_eval_suite.utils.plot_viewer import PlotViewer

"""
🧪 Champion Model Validation Dashboard

This module builds an interactive Jupyter dashboard using ipywidgets to visualize the
final holdout evaluation results of a production-ready model.

Key Features:
- KPI summary of metrics like accuracy, F1, ROC AUC, R-squared, RMSE, etc.
- Tabbed plot viewers for evaluation and feature distribution visualizations
- SHAP-based local explainer for interpretability on holdout predictions

Entrypoint:
- display_validation_dashboard(): renders the dashboard from the results dictionary
"""

def _build_html_card(title: str, content: str, style: str = "") -> widgets.HTML:
    """Helper function to create styled card widgets."""
    return widgets.HTML(f"""<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 8px 0; {style}">
                           <h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">{title}</h4>{content}</div>""")

class ValidationDashboard:
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.metrics = results.get("metrics", {})
        self.suite_config: Optional[SuiteConfig] = results.get("config_obj")
        self.widget = self._build_dashboard()

    def _build_summary_tab(self) -> widgets.VBox:
        """Dynamically builds a summary tab by searching for the exact metric keys produced by the evaluation engine."""
        kpi_html_parts = []

        # Define supported metrics and key variations
        possible_metrics = [
            {'label': 'Accuracy', 'keys': ['accuracy', 'final_accuracy', 'test_accuracy'], 'format': '.3f'},
            {'label': 'F1-Score', 'keys': ['f1_score', 'final_f1_score', 'test_f1'], 'format': '.3f'},
            {'label': 'Precision', 'keys': ['precision', 'final_precision', 'test_precision'], 'format': '.3f'},
            {'label': 'Recall', 'keys': ['recall', 'final_recall', 'test_recall'], 'format': '.3f'},
            {'label': 'ROC AUC', 'keys': ['roc_auc', 'test_roc_auc'], 'format': '.3f'},
            {'label': 'R-squared', 'keys': ['r_squared', 'final_r_squared', 'test_r_squared'], 'format': '.3f'},
            {'label': 'RMSE', 'keys': ['root_mean_squared_error', 'final_rmse'], 'format': ',.2f'}, # ADDED
            {'label': 'MAE', 'keys': ['mean_absolute_error', 'final_mae'], 'format': ',.2f'}, # ADDED
        ]

        baseline_metrics = self.results.get("baseline_metrics", {})
        if baseline_metrics is None:
            baseline_metrics = {}

        for metric_info in possible_metrics:
            found_key = next((key for key in metric_info['keys'] if key in self.metrics), None)
            if found_key:
                value = self.metrics[found_key]
                baseline_val = next((baseline_metrics.get(key) for key in metric_info['keys'] if key in baseline_metrics), None)
                delta = value - baseline_val if baseline_val is not None else None
                card_html = f"""<div style="text-align: center;">
                    <p style="font-size: 14px; margin: 0; color: #586069;">{metric_info['label']}</p>
                    <p style="font-size: 24px; font-weight: bold; margin: 0; color: #0366d6;">{value:{metric_info['format']}}</p>
                """

                # Show delta vs baseline for all supported metrics (single line, no inner loop)
                if baseline_val is not None and delta is not None:
                    sign = "+" if delta >= 0 else "-"
                    color = "green" if delta >= 0 else "red"
                    delta_str = f"{sign}{abs(delta):.3f}"
                    card_html += f"""<p style="font-size: 12px; margin: 0; color: {color};">Δ vs baseline: {delta_str}</p>"""

                std_key = f"{found_key}_std"
                std_val = self.metrics.get(std_key)
                if std_val is not None:
                    card_html += f"""<p style="font-size: 12px; margin: 0; color: #6a737d;">± {std_val:.3f}</p>"""

                card_html += "</div>"
                kpi_html_parts.append(card_html)

        if not kpi_html_parts:
            return widgets.VBox([_build_html_card("Final Holdout Metrics", "<i>No summary metrics found in results.</i>")])

        kpi_html = f'<div style="display: flex; justify-content: space-around;">{"".join(kpi_html_parts)}</div>'

        return widgets.VBox([_build_html_card("Final Holdout Metrics", kpi_html)])

    def _build_plot_viewer_tab(self, title: str, sub_directory: str) -> widgets.VBox:
        if not self.suite_config:
            return widgets.VBox([widgets.HTML("<i>Configuration object not found.</i>")])

        # Locate directory for saved plots
        plot_dir = self.suite_config.paths.plots_dir / self.suite_config.run_id / sub_directory
        
        # Handle missing or empty plot directories
        if not plot_dir.is_dir() or not any(p.suffix.lower() in PlotViewer.SUPPORTED_FORMATS for p in plot_dir.iterdir()):
             return widgets.VBox([widgets.HTML(f"<p><em>No plot images found in '{sub_directory}'.</em></p>")])
        
        # Initialize viewer widget
        viewer = PlotViewer(str(plot_dir), title=title)
        return widgets.VBox([viewer.widget_box])

    def _build_explainer_tab(self) -> widgets.VBox:
        # Retrieve SHAP objects and input data
        explainer = self.results.get('explainer')
        data_for_shap = self.results.get('data_for_shap')
        shap_values_obj = self.results.get('shap_values')

        # Skip if SHAP output is unavailable
        if not all([explainer, data_for_shap is not None, shap_values_obj is not None]):
            return widgets.VBox([widgets.HTML("<i>SHAP analysis was not run for this model.</i>")])

        shap.initjs()
        description = widgets.HTML("<p>Enter a row index to explain its prediction on the holdout set.</p>")
        index_input = widgets.IntText(value=0, description='Row Index:', style={'description_width': 'initial'})
        explain_button = widgets.Button(description="Explain Prediction")
        interactive_controls = widgets.HBox([index_input, explain_button])
        output_area = widgets.Output()

        # Define SHAP prediction callback
        def on_explain_click(b):
            with output_area:
                output_area.clear_output(wait=True)
                idx = index_input.value
                if not 0 <= idx < len(data_for_shap):
                    print(f"Error: Index must be between 0 and {len(data_for_shap) - 1}")
                    return
                
                # Render SHAP force plot
                force_plot = shap.force_plot(
                    explainer.expected_value, 
                    shap_values_obj.values[idx], 
                    data_for_shap.iloc[idx, :], 
                    matplotlib=False
                )
                display(IHTML(force_plot.html()))

        explain_button.on_click(on_explain_click)
        
        title = widgets.HTML('<h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">SHAP Local Explainer</h4>')
        content = widgets.VBox([description, interactive_controls, output_area])
        return widgets.VBox([title, content], layout=widgets.Layout(
            border='1px solid #e1e4e8', border_radius='6px', padding='16px', margin='8px 0'
        ))

    def _build_baseline_comparison_tab(self) -> widgets.VBox:
        baseline_metrics = self.results.get("baseline_metrics")
        if not baseline_metrics:
            return widgets.VBox([widgets.HTML("<i>No baseline metrics available for comparison.</i>")])

        items = []
        for key, val in baseline_metrics.items():
            value_str = f"{val:.3f}" if isinstance(val, float) else str(val)
            items.append(f"<tr><td style='padding:4px 12px;'><b>{key}</b></td><td style='padding:4px 12px;'>{value_str}</td></tr>")
        table_html = f"<table style='border-collapse:collapse;'>{''.join(items)}</table>"
        return widgets.VBox([_build_html_card("Baseline Model Metrics", table_html)])

    def _build_dashboard(self) -> widgets.VBox:
        if not self.suite_config:
            return widgets.VBox([widgets.HTML("<h2>Validation Run Failed: Configuration object not found in results.</h2>")])
        
        run_id = self.suite_config.run_id
        # Header banner
        banner = widgets.HTML(f"<h2>Champion Model Validation: {run_id}</h2>")

        # Assemble dashboard tabs
        tab_children = [
            self._build_summary_tab(),
            self._build_plot_viewer_tab(title="Evaluation Plots", sub_directory="evaluation"),
            self._build_plot_viewer_tab(title="Feature Distributions", sub_directory="distributions"),
            self._build_explainer_tab()
        ]
        
        tab_titles = ["📊 Summary", "🖼️ Evaluation Plots", "📈 Distributions", "🧠 Explainability"]

        if self.results.get("baseline_metrics"):
            tab_children.append(self._build_baseline_comparison_tab())
            tab_titles.append("📉 Baseline Model Metrics")

        tabs = widgets.Tab(children=tab_children)
        # Name each tab with emojis
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
            
        # Final dashboard layout
        return widgets.VBox([banner, tabs])

    def display(self):
        display(self.widget)

def display_validation_dashboard(results: Dict[str, Any]):
    """Public function to instantiate and display the dashboard."""
    # Entry point for rendering
    dashboard = ValidationDashboard(results)
    dashboard.display()