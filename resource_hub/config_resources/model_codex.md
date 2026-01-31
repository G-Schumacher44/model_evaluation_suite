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

## 🧰 Config Codex: Which Model Name Runs What?

These are the supported models in the `model_eval_suite` and how to configure them in your YAML file.

Each row tells you:
- The **pipeline factory name** to use under `pipeline_factory.name`
- The corresponding `run_to_execute` value
- The task type (`classification` or `regression`)

### 🔍 Classification Models

| Model Description         | `pipeline_factory.name` | `run_to_execute`                     |
| ------------------------- | ----------------------- | ------------------------------------ |
| Logistic Regression       | `"LogisticRegression"`  | `logistic_regression_classifier_run` |
| Random Forest Classifier  | `"RandomForest"`        | `random_forest_classifier_run`       |
| XGBoost Classifier        | `"XGBoost"`             | `xgboost_classifier_run`             |
| Support Vector Classifier | `"SVC"`                 | `svc_classifier_run`                 |
| Gaussian Naive Bayes      | `"GaussianNB"`          | `gaussian_nb_classifier_run`         |
| Decision Tree Classifier  | `"DecisionTree"`        | `decision_tree_classifier_run`       |

### 📈 Regression Models

| Model Description        | `pipeline_factory.name`   | `run_to_execute`              |
| ------------------------ | ------------------------- | ----------------------------- |
| Linear Regression        | `"LinearRegression"`      | `linear_regression_run`       |
| Random Forest Regressor  | `"RandomForestRegressor"` | `random_forest_regressor_run` |
| XGBoost Regressor        | `"XGBRegressor"`          | `xgboost_regressor_run`       |
| Decision Tree Regressor  | `"DecisionTreeRegressor"` | `decision_tree_regressor_run` |
| Support Vector Regressor | `"SVR"`                   | `svr_regressor_run`           |

---

### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)
