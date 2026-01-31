<p align="center">
  <img src="../repo_files/hero_banner.png" width="600"/>
  <br>
  <em>Model Evaluation + Interpretability Engine</em>
</p>
<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-yellow">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blueviolet">
</p>

This hub serves as the main navigation portal for configuration, modeling, and evaluation documentation.

## 🗂️ Resource Hub Structure

```
resource_hub/
├── config_resources/
│   ├── config_guide.md
│   ├── model_factory_params.md
│   ├── model_codex.md
│   ├── annotated_example_yaml/
│   ├── template_yaml/
│   └── config.zip
├── modeling_resources/
│   ├── feature_engineering.md
│   └── MLFlow.md
├── notebook_resources/
│   └── notebook_walkthrough.md
├── CLI_resources/
│   └── cli_usage_guide.md
└── hub_index.md

```

---

## 📚 Demo & Quick Start Notebooks

<p align="left">
  <a href="../notebooks/demo.ipynb">
    <img alt="Demo Notebook" src="https://img.shields.io/badge/Demo%20Notebook-blue?style=for-the-badge&logo=jupyter" />
  </a>
  &nbsp;&nbsp;
  <a href="../notebooks/quick_start.ipynb">
    <img alt="Quick Start Notebook" src="https://img.shields.io/badge/Quick--Start%20Notebook-green?style=for-the-badge&logo=python" />
  </a>
</p>

___

### 🔗 Sitemap & Resource Categories

#### 🧩 Configuration Resources
- [Configuration Guide](config_resources/config_guide.md) — YAML structure, templates, and examples
- [Model Factory Parameters](config_resources/model_factory_params.md) — Valid model names and hyperparameters
- [Model Codex](config_resources/model_codex.md) — Supported estimators and pipeline behavior
- [Annotated Config Examples](config_resources/annotated_example_yaml) — Explainitory YAML files
- [General Template Config](config_resources/template_yaml) — YAML templates
- [YAML Template Bundle](config_resources/config.zip) — Downloadable starter config pack

#### ⚙️ Modeling Resources
- [Feature Engineering Guide](modeling_resources/feature_engineering.md) — How to plug in your own custom transformer
- [MLflow Integration Guide](modeling_resources/MLFlow.md) — Setup, model registration, and tracking

#### 📓 Notebook Resources
- [Notebook Walkthrough](notebook_resources/notebook_walkthrough.md) — End-to-end workflow guide for Jupyter Notebooks
- [Sample Reports & Artifacts](../exports/sample/): Example outputs from live runs — includes QA reports, validation logs, feature plots, and dashboard visuals
- [notebooks/demo.ipynb](../notebooks/demo.ipynb): Interactive walkthrough showcasing the evaluation suite’s capabilities
  - [notebooks/demo_notebook.pdf](../notebooks/demo_notebook.pdf): viewable PDF version of the demo notebook as a pdf file
  - [notebooks/demo_notebook.html](../notebooks/demo_notebook.html): download the demo notebook as a html file


#### 🖥️ CLI Resources
- [CLI Usage Guide](CLI_resources/cli_usage_guide.md) — End-to-end workflow in command-line.

---

> Return to the top-level project overview: [Main Repository README](../README.md)
