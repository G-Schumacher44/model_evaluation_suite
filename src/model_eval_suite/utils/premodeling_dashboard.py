"""
🧪 Pre-Model Diagnostics Dashboard (Jupyter Widget)

This module provides an interactive HTML-styled dashboard to visualize the results of pre-model
diagnostics such as missingness, skewness, outliers, and collinearity (VIF).

Responsibilities:
- Render data quality summaries and tables in separate tabs
- Display diagnostic plots using the embedded PlotViewer utility
- Provide a clean and collapsible overview for quick inspection

Entrypoint:
- display_pre_modeling_dashboard(): builds and displays the dashboard from config + diagnostics
"""
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from pathlib import Path
from typing import Dict, Any

from .plot_viewer import PlotViewer

def _build_html_card(title: str, content: str, style: str = "") -> widgets.HTML:
    """Helper function to create styled card widgets."""
    return widgets.HTML(f"""<div style="border: 1px solid #e1e4e8; border-radius: 6px; padding: 16px; margin: 8px 0; {style}">
                           <h4 style="margin-top:0; border-bottom: 1px solid #e1e4e8; padding-bottom: 8px;">{title}</h4>{content}</div>""")

class PreModelingDashboard:
    def __init__(self, diagnostic_results: Dict[str, pd.DataFrame], config: Any):
        self.diagnostic_results = diagnostic_results
        self.config = config
        self.widget = self._build_dashboard()

    def _build_overview_tab(self) -> widgets.VBox:
        """Builds a high-level summary of the diagnostic findings."""
        # Build a bulleted summary based on diagnostics results
        items = []
        
        # Summarize Missingness
        missing_df = self.diagnostic_results.get("missingness_summary")
        if missing_df is not None and not missing_df.empty:
            items.append(f"<li><b>{len(missing_df)}</b> columns have missing values.</li>")

        # Summarize Skewness
        skew_df = self.diagnostic_results.get("skewness_summary")
        if skew_df is not None and not skew_df.empty:
            items.append(f"<li><b>{len(skew_df)}</b> numeric features are highly skewed.</li>")

        # Summarize Outliers
        outlier_df = self.diagnostic_results.get("outlier_summary")
        if outlier_df is not None and not outlier_df.empty:
            items.append(f"<li><b>{len(outlier_df)}</b> numeric features have potential outliers.</li>")
            
        # Summarize VIF
        vif_df = self.diagnostic_results.get("vif_scores")
        vif_threshold = self.config.pre_model_diagnostics.vif_threshold
        if vif_df is not None and not vif_df.empty:
            high_vif_count = len(vif_df[vif_df['VIF'] > vif_threshold])
            if high_vif_count > 0:
                items.append(f"<li><b>{high_vif_count}</b> features have high multicollinearity (VIF > {vif_threshold}).</li>")

        if not items:
            summary_html = "<p>✅ No major data quality issues were detected based on the current thresholds.</p>"
        else:
            summary_html = "<ul>" + "".join(items) + "</ul>"
            
        return widgets.VBox([_build_html_card("Diagnostics Overview", summary_html)])

    def _build_report_panes(self):
        """Returns a list of individual diagnostic report widgets and their titles."""
        # Generate widgets for each diagnostic category (Missing, VIF, Skew, Outliers)
        panes = []
        titles = []

        # Overview
        overview_pane = self._build_overview_tab()
        panes.append(overview_pane)
        titles.append("📋 Overview")

        # Tab: Missingness summary table
        missing_df = self.diagnostic_results.get("missingness_summary")
        if missing_df is not None and not missing_df.empty:
            pane = _build_html_card("Missing Values", missing_df.to_html(classes='table'))
        else:
            pane = widgets.HTML("No missing values found.")
        panes.append(pane)
        titles.append("🧩 Missingness")

        # Tab: VIF scores table
        vif_df = self.diagnostic_results.get("vif_scores")
        if vif_df is not None and not vif_df.empty:
            pane = _build_html_card("VIF Scores", vif_df.to_html(classes='table'))
        else:
            pane = widgets.HTML("VIF scores were not calculated.")
        panes.append(pane)
        titles.append("🔁 Collinearity")

        # Tab: Distribution summaries for skewness and outliers
        skew_df = self.diagnostic_results.get("skewness_summary")
        outlier_df = self.diagnostic_results.get("outlier_summary")
        dist_pane = widgets.HBox([
            _build_html_card("Skewed Features", skew_df.to_html(classes='table')) if skew_df is not None and not skew_df.empty else widgets.HTML("<i>No highly skewed features found.</i>"),
            _build_html_card("Outlier Counts", outlier_df.to_html(classes='table')) if outlier_df is not None and not outlier_df.empty else widgets.HTML("<i>No outliers detected.</i>")
        ])
        panes.append(dist_pane)
        titles.append("📊 Distributions")

        return panes, titles

    def _build_plots_tab(self) -> widgets.VBox:
        """Builds a tab with a PlotViewer for diagnostic plots."""
        # Display diagnostic plots if available using PlotViewer
        plot_dir = self.config.paths.plots_dir / self.config.run_id / "diagnostics"
        if not plot_dir.is_dir() or not any(p.suffix.lower() in PlotViewer.SUPPORTED_FORMATS for p in plot_dir.iterdir()):
             return widgets.VBox([widgets.HTML(f"<p><em>No diagnostic plot images found.</em></p>")])
        viewer = PlotViewer(str(plot_dir), title="Diagnostic Plots")
        return widgets.VBox([viewer.widget_box])

    def _build_dashboard(self) -> widgets.Accordion:
        # Assemble all tabs into an Accordion container for interactive display
        report_panes, titles = self._build_report_panes()
        plots_tab_container = self._build_plots_tab()

        children = report_panes + [plots_tab_container]
        tab_widget = widgets.Tab(children=children)
        for idx, title in enumerate(titles + ['🖼️ Eval Plots']):
            tab_widget.set_title(idx, title)

        accordion = widgets.Accordion(children=[tab_widget])
        accordion.set_title(0, '🔍 Pre-Modeling Diagnostics')
        accordion.selected_index = None  # Start collapsed

        return accordion

    def display(self):
        """Renders the assembled widget."""
        # Render the dashboard widget if diagnostics are available
        if not self.diagnostic_results:
            return
        display(self.widget)

def display_pre_modeling_dashboard(diagnostic_results: Dict[str, Any], config: Any):
    # Entrypoint to build and display the diagnostics dashboard
    dashboard = PreModelingDashboard(diagnostic_results, config)
    dashboard.display()