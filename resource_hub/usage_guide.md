<file name=0 path=/Users/garrettschumacher/Documents/git_repos/model_evaluation_suite/README.md><p align="center">
  <img src="../repo_files/hero_banner.png" width="600"/>
  <br>
  <em>Model Evaluation + Interpretability Engine</em>
</p>
<p align="center">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-beta-yellow">
  <img alt="Version" src="https://img.shields.io/badge/version-v0.2.0-blueviolet">
</p>

# 🚀 Usage Guide: model_evaluation_suite

This document walks you through using the model evaluation suite from start to finish. It includes installation, configuration, running experiments, and validating production models.

---

## 1️⃣ Install Requirements

Make sure you have Python 3.9+ and the required dependencies:

**Option 1: Conda (Recommended)**

```bash
# Create and activate conda environment
conda create -n model_eval_suite python=3.9
conda activate model_eval_suite
```

**Insall pip packages**

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Select or Create a Config

Choose a model configuration from the examples or create your own override YAML.

### Use a flat template:
```bash
cp config_resources/template_yaml/template_modeling.yaml config/my_run.yaml
```

Edit the config to match your input file and desired model (e.g., RandomForest).

### Key Field:
```yaml
run_to_execute: "random_forest_classifier_run"
```


You can also browse:
- [`example_modeling.yaml`](config_resources/annotated_example_yaml/example_modeling.yaml)
- [`example_default_config.yaml`](config_resources/annotated_example_yaml/example_default_config.yaml)

---

## 🔍 Optional: Add Custom Feature Engineering

The pipeline supports modular feature engineering using your own Python class.

In your config YAML:
```yaml
feature_engineering:
  run: true
  module: my_package.my_feature_engineering
  class_name: MyFeatureEngineer
```

Your class should implement scikit-learn's `fit` / `transform` interface.

📄 See [`feature_engineering.md`](modeling_resources/feature_engineering.md) for examples and guidance.

---

## 3️⃣ Run the Modeling Pipeline

To train and evaluate your model:

```bash
python src/model_eval_suite/run_pipeline.py --config config/my_run.yaml
```

- All artifacts, reports, and plots will be saved to the folders specified in your config.
- To view the HTML dashboard, open the exported file from your `reports_dir`.

---

## 4️⃣ Validate a Production Model (Champion Check)

Use this to audit a registered model from MLflow or a saved pickle file:

```bash
python src/model_eval_suite/validate_champion.py --config config/my_validation.yaml
```

Required fields include:
- `mlflow_tracking_uri`
- `model_source.name`
- `model_source.version`
- `holdout_data_path`
- `target_column`

See the [validation config template](config_resources/annotated_example_yaml/example_validation_template.yaml).

---

## 5️⃣ Optional: Set Up MLflow Logging

See the [MLflow Integration Guide](modeling_resources/MLFlow.md) to:
- Start the MLflow tracking server
- Log models and metrics
- Register models to a local or remote registry

---

## 🔗 More Resources

 - [Configuration Tutorial](config_resources/config_guide.md)
 - [Model Factory Parameters](config_resources/model_factory_params.md)
 - [Model Codex](config_resources/model_codex.md)
 - [Feature Engineering Guide](modeling_resources/feature_engineering.md)
---

### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../README.md)
