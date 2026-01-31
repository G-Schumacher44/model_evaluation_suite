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


# 🧠 Custom Feature Engineering Guide

This guide explains how to inject your own feature engineering logic into the evaluation pipeline using a scikit-learn compatible transformer.

---

## 📌 Requirements

Your custom class must:

- Inherit from `BaseEstimator`, `TransformerMixin` (optional but recommended)
- Implement `.fit(X, y=None)` and `.transform(X)`

---

## ✅ Example

```python
# my_project/custom_features.py

from sklearn.base import BaseEstimator, TransformerMixin

class MyFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X["project_hours_interaction"] = (
            X["number_project"] * X["average_montly_hours"]
        )
        return X
```

---

## 🔧 YAML Usage

Reference it like this in your YAML:

```yaml
feature_engineering:
  run: true
  module: "my_project.custom_features"
  class_name: "MyFeatureTransformer"
```

The suite will dynamically import and attach your transformer as the first step of the pipeline.

---

## 🧪 Testing Tips

- Keep the transformer stateless or only derive stats from `X` to avoid leakage.
- Run the notebook in `notebook_mode: true` for inline debugging.
- Raise a clear error if a required column is missing for better traceability.

---

Want to contribute a reusable transformer? Fork this repo and submit a PR to `src/model_eval_suite/custom_transformers/`!

---

### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)
