<p align="center">
  <img src="../../repo_files/dark_logo_banner.png" width="1000"/>
  <br>
  <em>Model Evaluation + Interpretability Engine</em>
</p>
<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-yellow">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.1.0-blueviolet">
</p>

## 🖥️ CLI Usage Guide for model_eval_suite

This guide provides instructions for using the model_eval_suite from the command line interface (CLI). The CLI is ideal for automation, production workflows, or quick evaluations outside of Jupyter.

### 📦 Prerequisites

Ensure the environment is set up and activated:

```bash
conda activate model_eval_suite
```

If needed, create the environment first:

```bash
conda env create -f environment.yml
```

---

### 🗂️ Folder Structure Assumption

```text
model_evaluation_suite/
├── config/                      # Your YAML configs (can be renamed)
├── data/                        # Raw and processed datasets
├── exports/                     # Optional: Exported HTML, metrics, etc.
└── src/model_eval_suite/        # Core package modules
```

---

### ▶️ Run a Model Evaluation

```bash
python -m model_eval_suite.run_pipeline --config config/example_override.yaml
```

This command triggers the full evaluation pipeline using the provided YAML config.

### What It Does

- Loads and merges the default config with your override
- Runs pre-model diagnostics (if enabled)
- Trains and evaluates the model
- Saves plots, metrics

---

### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)