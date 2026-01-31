<file name=0 path=/Users/garrettschumacher/Documents/git_repos/model_evaluation_suite/README.md><p align="center">
  <img src="../../repo_files/hero_banner.png" width="600"/>
  <br>
  <em>Model Evaluation + Interpretability Engine</em>
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


# 🔌 MLflow Startup & Configuration Guide

This guide explains how to run MLflow locally and use it with the `model_evaluation_suite`.

---

## 🚀 Start the Tracking Server Locally

```bash
mlflow ui
```

To run on a custom port and allow access from other devices:

```bash
mlflow ui --port 5050 --host 0.0.0.0
```

- `--port 5050`: change to any open port
- `--host 0.0.0.0`: exposes the UI to your local network (e.g. browse from iPad)

---

## 📂 Backend Store: SQLite (Recommended)

**Starting in v0.2.0**, the suite uses SQLite as the default MLflow backend instead of the filesystem.

**Why SQLite?**

- ✅ Better query performance
- ✅ Proper ACID transactions
- ✅ Future-proof (filesystem backend is deprecated in MLflow 2.18+)
- ✅ Easier to backup (single `mlflow.db` file)

**Default Configuration:**

```yaml
mlflow_tracking_uri: "sqlite:///mlflow.db"
```

**To use filesystem (legacy):**

```yaml
mlflow_tracking_uri: "file:./mlruns"
```

**To use a remote server:**

```yaml
mlflow_tracking_uri: "http://localhost:5000"
```

---

## 💾 Default File Structure

By default:
- All logs go to: `./mlruns/`
- Each experiment has its own folder
- Each run is nested by ID inside the experiment folder

---

## 🧠 MLflow + model_evaluation_suite

This evaluation suite integrates with MLflow by:

- Logging models using `mlflow.sklearn.log_model(...)`
- Registering models using `mlflow.register_model(...)`
- Logging metrics, parameters, and artifacts from validation

### To enable this:

1. Start the MLflow UI as shown above.
2. Set your tracking URI in the environment or YAML config:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

Or in your validation YAML:
```yaml
mlflow_tracking_uri: "http://localhost:5000"
```

3. Then run:
```bash
python src/model_eval_suite/validate_champion.py --config config/your_validation.yaml
```

---

## 📁 Recommended Project Layout

```plaintext
.
├── mlruns/                  # Local experiment logs (gitignored)
├── models/                  # Exported models
├── config/                  # YAML configurations
├── data/                    # Input/holdout/test sets
└── src/                     # Source code
```

> ✅ Tip: Add `mlruns/` to `.gitignore` unless versioning experiment logs.

---

## ✅ Quick Setup Checklist

- Install MLflow: `pip install mlflow`
- Start the tracking server: `mlflow ui`
- Set `MLFLOW_TRACKING_URI` in env or YAML
- Run `validate_champion.py` to log model + metrics

---

## 🧠 Advanced (Optional)

You can point to a remote tracking server or use a cloud artifact store:

```bash
export MLFLOW_TRACKING_URI="http://mlflow.yourdomain.com"
export MLFLOW_S3_ENDPOINT_URL="https://s3.amazonaws.com"
```

---

#### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)
