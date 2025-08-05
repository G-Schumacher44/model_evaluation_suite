import ipywidgets as widgets
import pandas as pd
import json
import shap
import numpy as np
import matplotlib.pyplot as plt
import io
import seaborn as sns
from IPython.display import display, HTML as IHTML
from pathlib import Path
from typing import Optional, Dict, Any

from model_eval_suite.utils.plot_viewer import PlotViewer
from model_eval_suite.utils.config import SuiteConfig
from sklearn.linear_model import LinearRegression


"""
🖥️ Regression Dashboard for model evaluation.

This dashboard provides an interactive summary of regression model results.
Tabs include:
- Summary KPIs and fit statistics
- Static importance plots (coefficients, permutation, SHAP)
- SHAP-based local explainability
- Evaluation and distribution plots (via PlotViewer)
- Text-based statistical summaries (statsmodels)
- Full config display for reproducibility

Entrypoint:
- display_regression_dashboard(results): renders the full interactive dashboard
"""

def _build_html_card(title: str, content: str, style: str = "") -> widgets.HTML:
    return widgets.HTML(f"""<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 8px 0; {style}">
                           <h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">{title}</h4>{content}</div>""")

class RegressionEvaluationDashboard:
    def __init__(self, results: Dict[str, Any]):
        # Extract core components from results dictionary
        self.results = results
        self.config: SuiteConfig = results.get("config_obj")
        self.metrics = results.get("metrics", {})
        self.plot_paths = results.get("plot_paths", {})
        self.final_model = results.get("final_model")
        self.baseline_metrics = self.results.get("baseline_metrics")
        self.widget = self._build_dashboard()

    def _find_baseline_value(self, metric_keys: list) -> float | None:
        """Searches for the first available metric key in the baseline metrics."""
        for key in metric_keys:
            if key in self.baseline_metrics:
                return self.baseline_metrics[key]
        return None

    def _render_banner(self) -> widgets.HTML:
        # Render a header banner showing run ID and R-squared score
        r2 = self.metrics.get('r_squared', 0)
        run_id = self.config.run_id if self.config else "N/A"
        return widgets.HTML(f"""<div style="border: 1px solid #d0d7de; background-color: #f6f8fa; padding: 12px; border-radius: 6px; margin-bottom: 1em;">
                                <strong>Run:</strong> {run_id} | <strong>✅ R-squared:</strong> {r2:.3f}
                            </div>""")
    def _build_importance_tab(self) -> widgets.VBox:
        """Displays static importance plots like coefficients and permutation importance."""
        items = []
        
        # Display available static feature importance plots (if they exist)
        importance_plot_keys = [
            'feature_coefficients', # For Linear Models
            'permutation_importance', # For all models
            'shap_bar_plot' # For all models where SHAP is run
        ]
        
        for key in importance_plot_keys:
            plot_path_str = self.plot_paths.get(key)
            if plot_path_str and Path(plot_path_str).exists():
                image_bytes = Path(plot_path_str).read_bytes()
                if image_bytes:  # Only attempt to render if file is not empty
                    title = key.replace('_', ' ').replace('Plot', '').strip().title()
                    image_widget = widgets.Image(value=image_bytes, format='png', width=700)
                    items.append(widgets.VBox([widgets.HTML(f"<h4>{title}</h4>"), image_widget]))
        
        if not items:
            return widgets.VBox([widgets.HTML("<i>No importance plots were generated for this run.</i>")])
        return widgets.VBox(items)

    
    def _build_summary_tab(self) -> widgets.VBox:
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

        kpi_html_parts = []
        possible_metrics = [
            {'label': 'R-squared', 'keys': ['r_squared', 'test_r_squared'], 'format': '.3f'},
            {'label': 'RMSE', 'keys': ['root_mean_squared_error', 'test_rmse'], 'format': ',.2f'},
            {'label': 'MAE', 'keys': ['mean_absolute_error', 'test_mae'], 'format': ',.2f'}
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

        return widgets.VBox([
            _build_html_card("Test Set Performance", f"<div style='display: flex; justify-content: space-around;'>{''.join(kpi_html_parts)}</div>"),
            cv_plot_card
        ])

    def _build_explainer_tab(self) -> widgets.VBox:
        """This tab is dedicated to interactive SHAP analysis."""
        explainer = self.results.get('explainer')
        data_for_shap = self.results.get('data_for_shap')
        shap_values_obj = self.results.get('shap_values')

        # Exit early if SHAP artifacts are missing
        if not all([explainer, data_for_shap is not None, shap_values_obj is not None]):
            return widgets.VBox([widgets.HTML("<i>SHAP analysis was not run for this model.</i>")])

        shap.initjs()
        description = widgets.HTML("<p>Enter a row index to explain its prediction.</p>")
        index_input = widgets.IntText(value=0, description='Row Index:', style={'description_width': 'initial'})
        explain_button = widgets.Button(description="Explain Prediction")
        interactive_controls = widgets.HBox([index_input, explain_button])
        output_area = widgets.Output()

        # Define callback to display SHAP force plot for selected row
        def on_explain_click(b):
            with output_area:
                output_area.clear_output(wait=True)
                idx = index_input.value
                if not 0 <= idx < len(data_for_shap):
                    print(f"Error: Index must be between 0 and {len(data_for_shap) - 1}")
                    return
                
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

    def _build_reports_tab(self) -> widgets.VBox:
        # Render plain-text statsmodels report, if generated
        summary_path_str = self.plot_paths.get('statistical_summary')
        if summary_path_str and Path(summary_path_str).exists():
            summary_text = Path(summary_path_str).read_text()
            return _build_html_card("Statistical Summary", f"<pre>{summary_text}</pre>")
        return widgets.VBox([widgets.HTML("<i>No text reports were generated for this run.</i>")])

    def _build_plot_viewer_tab(self, title: str, sub_directory: Optional[str] = None) -> widgets.VBox:
        """Builds a tab content using the PlotViewer widget."""
        if not self.config: return widgets.VBox([widgets.HTML("<i>Configuration not available.</i>")])
        
        # Construct plot directory path based on config and tab
        plot_dir = self.config.paths.plots_dir / self.config.run_id
        if sub_directory:
            plot_dir = plot_dir / sub_directory
        # ------------------------------------

        if not plot_dir.is_dir() or not any(p.suffix.lower() in PlotViewer.SUPPORTED_FORMATS for p in plot_dir.iterdir()):
             return widgets.VBox([widgets.HTML(f"<p><em>No plot images found in '{sub_directory or 'main plots'}'.</em></p>")])
        
        viewer = PlotViewer(str(plot_dir), title=title)
        return widgets.VBox([viewer.widget_box])
    
    def _build_config_tab(self) -> widgets.HTML:
        # Display full config JSON used for the run
        if not self.config:
            return widgets.HTML("<i>Configuration not available.</i>")
        
        config_str = self.config.model_dump_json(indent=2)
        return widgets.HTML(f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{config_str}</pre>")

    def _render_audit_alerts(self) -> widgets.HTML:
        # Show any warnings or audit alerts flagged during model evaluation
        alerts = self.results.get('audit_alerts', [])
        if not alerts:
            return widgets.HTML("")
        items = "".join([f"<li>{alert}</li>" for alert in alerts])
        return widgets.HTML(f"""<div style="border: 1px solid #f0b37e; background: #fff2e5; padding: 12px; border-radius: 6px; margin-top:1em;">
                                <strong>Audit Alerts:</strong><ul>{items}</ul>
                            </div>""")

    def _build_dashboard(self) -> widgets.VBox:
        if not self.config:
            return widgets.VBox([widgets.HTML("Run failed: Configuration object not found.")])
        banner = self._render_banner()

        # Start with summary tab
        tab_children = [
            self._build_summary_tab(),
        ]
        tab_titles = [
            '📊 Summary',
        ]

        # Conditionally add the baseline tab
        if self.baseline_metrics:
            tab_children.append(self._build_baseline_tab())
            tab_titles.append('📉 Baseline Comparison')

        # Continue adding other tabs
        tab_children += [
            self._build_importance_tab(),
            self._build_explainer_tab(),
            self._build_plot_viewer_tab(title="Evaluation Plots", sub_directory="evaluation"),
            self._build_plot_viewer_tab(title="Feature Distributions", sub_directory="distributions"),
            self._build_reports_tab(),
            self._build_config_tab()
        ]
        tab_titles += [
            '🎯 Importance',
            '🧠 Explainability',
            '🖼️ Eval Plots',
            '📈 Distributions',
            '📝 Reports',
            '⚙️ Configuration'
        ]

        main_tabs = widgets.Tab(children=tab_children)
        for i, title in enumerate(tab_titles):
            main_tabs.set_title(i, title)
        return widgets.VBox([banner, self._render_audit_alerts(), main_tabs])

    def _build_baseline_tab(self) -> widgets.HTML:
        if not self.baseline_metrics:
            return widgets.HTML("<i>No baseline run specified.</i>")

        baseline_id = self.baseline_metrics.get('run_id', 'Baseline')
        metrics_to_compare = ['r_squared', 'root_mean_squared_error', 'mean_absolute_error']
        data = []

        for m in metrics_to_compare:
            if m in self.metrics and m in self.baseline_metrics:
                current_val = self.metrics[m]
                baseline_val = self.baseline_metrics[m]
                delta = current_val - baseline_val
                row = {
                    'Metric': m.replace('_', ' ').title(),
                    'Current Model': f"{current_val:.3f}",
                    f"Baseline ({baseline_id})": f"{baseline_val:.3f}",
                    "Change": f"{delta:+.3f}"
                }
                data.append(row)

        if not data:
            return widgets.HTML("<i>No overlapping metrics available for comparison.</i>")

        df = pd.DataFrame(data)
        content = df.to_html(index=False, classes='table', border=0)
        return _build_html_card(f"Comparison to Baseline ({baseline_id})", content)

    def display(self):
        display(self.widget)

def display_regression_dashboard(results: Dict[str, Any]):
    dashboard = RegressionEvaluationDashboard(results)
    dashboard.display()