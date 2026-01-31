<p align="center">
  <img src="../../repo_files/hero_banner.png" width="600"/>
  <br>
  <em>Model Evaluation + Interpretability Engine</em>
</p>
<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-yellow">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blueviolet">
</p>

## 🖥️ CLI Usage Guide

The Model Evaluation Suite provides a modern, user-friendly command-line interface powered by Click and Rich for beautiful terminal output.

### 📦 Prerequisites

Install the suite in development mode:

```bash
pip install -e .
```

Or if using conda:

```bash
conda activate model_eval_suite
pip install -e .
```

---

## 🎯 Quick Start

```bash
# Get help (use --help or -h)
model-eval --help
model-eval -h

# Generate a new config
model-eval init --task classification

# Run evaluation
model-eval run config/my_model.yaml

# Prepare data
model-eval prep config/data_prep.yaml

# Validate model
model-eval validate config/validation.yaml

# List available models
model-eval list-models
```

---

## 📋 Command Reference

### `model-eval run`

Run the complete model evaluation pipeline.

```bash
model-eval run CONFIG_PATH [OPTIONS]

Options:
  --notebook-mode / --no-notebook-mode  Enable notebook mode (default: False)
  --logging [auto|on|off]               Logging mode (default: auto)
  -h, --help                            Show help message
```

**Examples:**
```bash
# Basic usage
model-eval run config/xgboost.yaml

# With logging enabled
model-eval run config/logreg.yaml --logging on

# Notebook mode (suppresses console output)
model-eval run config/model.yaml --notebook-mode
```

**What it does:**
- Loads configuration and merges with defaults
- Runs pre-model diagnostics (if enabled)
- Trains model with optional hyperparameter tuning
- Generates evaluation metrics and plots
- Creates SHAP explanations
- Logs to MLflow
- Generates audit alerts
- Exports HTML dashboards

---

### `model-eval init`

Generate a new configuration file template.

```bash
model-eval init [OPTIONS]

Options:
  --task [classification|regression]  Type of ML task (required)
  --output PATH                       Output path (default: config/my_config.yaml)
  --template [minimal|full]           Template type (default: minimal)
  -h, --help                          Show help message
```

**Examples:**
```bash
# Interactive prompt for task type
model-eval init

# Generate classification config
model-eval init --task classification --output config/clf.yaml

# Generate full regression template
model-eval init --task regression --template full
```

**What it creates:**
- Ready-to-use YAML configuration
- Sensible defaults for the task type
- Comments and examples
- Proper structure for all config sections

---

### `model-eval prep`

Prepare and split dataset for training.

```bash
model-eval prep CONFIG_PATH

Arguments:
  CONFIG_PATH  Path to data preparation config file
```

**Examples:**
```bash
model-eval prep config/data_prep.yaml
```

**What it does:**
- Loads raw data from CSV
- Performs train/test split with stratification
- Saves split datasets
- Validates data schema

---

### `model-eval validate`

Validate a champion model against holdout data.

```bash
model-eval validate CONFIG_PATH [OPTIONS]

Options:
  --model-name TEXT  Model name to validate
  --version TEXT     Model version
  -h, --help         Show help message
```

**Examples:**
```bash
# Basic validation
model-eval validate config/validation.yaml

# Validate specific model version
model-eval validate config/val.yaml --model-name xgb_model --version 1
```

**What it does:**
- Loads champion model from MLflow
- Evaluates on holdout/validation data
- Compares to baseline metrics
- Generates validation report
- Creates comparison plots

---

### `model-eval list-models`

Show all available model types.

```bash
model-eval list-models
```

**Output:**
- Table of classification models
- Table of regression models
- Config names for each model type
- Brief descriptions

---

## 🗂️ Project Structure

```text
model_evaluation_suite/
├── config/                      # Your YAML configurations
│   ├── classifier/              # Classification configs
│   ├── regressor/               # Regression configs
│   └── validation/              # Validation configs
├── data/                        # Raw and processed datasets
├── exports/                     # Output artifacts
│   ├── plots/                   # Visualization outputs
│   ├── reports/                 # HTML dashboards
│   └── models/                  # Serialized models
├── logs/                        # Execution logs
└── mlflow.db                    # MLflow tracking database
```

---

## 💡 Tips & Best Practices

**1. Start with a template:**
```bash
model-eval init --task classification
# Edit the generated config
model-eval run config/my_config.yaml
```

**2. Check available models:**
```bash
model-eval list-models
```

**3. Use meaningful run IDs:**
```yaml
run_id: "xgboost_feature_eng_v2"  # Good
run_id: "test"                     # Bad
```

**4. Enable logging for debugging:**
```bash
model-eval run config.yaml --logging on
```

**5. Review MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## 🔧 Troubleshooting

**Command not found:**
```bash
# Reinstall in development mode
pip install -e .
```

**Import errors:**
```bash
# Install all dependencies
pip install -e ".[dev]"
```

**Config validation errors:**
```bash
# Use init to generate a valid template
model-eval init --task classification
```

---

## 📎 Quick Links

- [Resource Hub Index](../hub_index.md)
- [Main README](../../README.md)
- [Config Guide](../config_resources/config_guide.md)
- [Notebook Walkthrough](../notebook_resources/notebook_walkthrough.md)
