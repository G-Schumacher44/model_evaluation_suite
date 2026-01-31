<file name=0 path=/Users/garrettschumacher/Documents/git_repos/model_evaluation_suite/README.md><p align="center">
  <img src="../../repo_files/hero_banner.png" width="600"/>
  <br>
  <em>Notebook Workflow Overview</em>
</p>
<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-yellow">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blueviolet">
  <br>
  <a href="https://github.com/G-Schumacher44/model_evaluation_suite/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/G-Schumacher44/model_evaluation_suite/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/G-Schumacher44/model_evaluation_suite">
    <img alt="Coverage" src="https://codecov.io/gh/G-Schumacher44/model_evaluation_suite/branch/main/graph/badge.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue">
  <img alt="Tests" src="https://img.shields.io/badge/tests-70%20passing-success">
</p>


# Model Evaluation Suite – Demo Notebook Guide

This notebook demonstrates the full workflow of the Model Evaluation Suite using the `salifort_50k` dataset.

It walks through:
- Data preparation (train/test/holdout splits)
- Pre-model diagnostics (VIF, skew, outliers)
- Training multiple models via YAML configuration
- MLflow tracking
- Interactive dashboard rendering
- Champion model validation using a holdout set

## 📚 Demo & Quick Start Notebooks

<p align="left">
  <a href="../../notebooks/demo.ipynb">
    <img alt="Demo Notebook" src="https://img.shields.io/badge/Demo%20Notebook-blue?style=for-the-badge&logo=jupyter" />
  </a>
  &nbsp;&nbsp;
  <a href="../../notebooks/quick_start.ipynb">
    <img alt="Quick Start Notebook" src="https://img.shields.io/badge/Quick--Start%20Notebook-green?style=for-the-badge&logo=python" />
  </a>
</p>

## Requirements

Before running:
- Activate the suite’s environment
- Ensure your config files are set (see `config_resources/`)
- Ensure data paths in your YAMLs point to valid CSVs

## Key Notebook Sections

1. **Configuration Setup**  
   Load YAMLs that define models, paths, diagnostics, and tracking.

2. **Data Preparation**  
   Use the `prep_data()` function to split raw CSVs if needed.

3. **Experiment Runs**  
   Call `run_experiment()` with various model YAMLs.

4. **Validation Runs**  
   Use `validate_champion()` to evaluate and optionally benchmark champion models.

5. **Interactive Dashboards**  
   Visual diagnostics for model performance, explainability, and data quality.

---

## Reference Configs

| Stage             | Config File                                                    |
| ----------------- | -------------------------------------------------------------- |
| Data Prep         | `config/data_prep.yaml`                                        |
| Logistic Model    | `config/classifier/logreg.yaml`                                |
| XGBoost Regressor | `config/regressor/xgboost_reg.yaml`                            |
| Validation        | `config/xgb_validation.yaml`, `config/xgb_reg_validation.yaml` |

See [docs/feature_engineering.md](../modeling_resources/feature_engineering.md) for how to inject custom transformers.
See [config_resources](../config_resources/config_guide.md) for YAML configuration help.

---

## Outputs

- Artifacts saved under `plots/`, `reports/`, `logs/`
- Models registered to MLflow (if enabled)
- Dashboards rendered inline or saved

📎 [`View Sample Reports & Artifacts`](../exports/sample/)

---

### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)
