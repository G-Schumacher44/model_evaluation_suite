<p align="center">
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

# ⚙️ Model Factory Parameter Reference

Each model factory corresponds to a scikit-learn or XGBoost estimator.  
Use the `pipeline_factory.name` field to select the estimator, and override only the parameters you need.

> ℹ️ Each run key listed below (e.g., `random_forest_classifier_model`) corresponds directly to a model configuration anchor defined in the `example_default_config.yaml` file. These are referenced using YAML merge syntax (e.g., `pipeline_factory: *random_forest_classifier_model`) within a runnable experiment block.

This guide aligns with the definitions in `default_config_example.yaml`.

---
## 🧪 Classification Models

<details>
<summary>click to expand section</summary>

___

### 🔹 `"RandomForest"`

**Library**: `sklearn.ensemble.RandomForestClassifier`  
**Run key**: `random_forest_classifier_model`

```yaml
pipeline_factory:
  name: "RandomForest"
```

**Default Parameters:**

| Param              | Description              | Default Value |
| ------------------ | ------------------------ | ------------- |
| `n_estimators`     | Number of trees          | `100`         |
| `max_depth`        | Max depth of trees       | `12`          |
| `min_samples_leaf` | Minimum samples per leaf | `2`           |
| `class_weight`     | Class balancing          | `'balanced'`  |
| `random_state`     | Random seed              | `42`          |

---

### 🔹 `"LogisticRegression"`

**Library**: `sklearn.linear_model.LogisticRegression`  
**Run key**: `logistic_regression_classifier_model`

```yaml
pipeline_factory:
  name: "LogisticRegression"
```

**Default Parameters:**

| Param          | Description                       | Default Value |
| -------------- | --------------------------------- | ------------- |
| `penalty`      | Regularization type               | `'l2'`        |
| `solver`       | Optimization algorithm            | `'liblinear'` |
| `class_weight` | Class balancing                   | `'balanced'`  |
| `C`            | Regularization strength (inverse) | `1.0`         |
| `random_state` | Random seed                       | `42`          |

---

### 🔹 `"XGBoost"`

**Library**: `xgboost.XGBClassifier`  
**Run key**: `xgboost_classifier_model`

```yaml
pipeline_factory:
  name: "XGBoost"
```

**Default Parameters:**

| Param           | Description                        | Default Value       |
| --------------- | ---------------------------------- | ------------------- |
| `n_estimators`  | Number of boosting rounds          | `100`               |
| `max_depth`     | Tree depth                         | `5`                 |
| `learning_rate` | Step size shrinkage                | `0.1`               |
| `objective`     | Loss function                      | `'binary:logistic'` |
| `eval_metric`   | Metric to evaluate during training | `'logloss'`         |
| `random_state`  | Reproducibility                    | `42`                |

---

### 🔹 `"SVC"`

**Library**: `sklearn.svm.SVC`  
**Run key**: `svc_classifier_model`

```yaml
pipeline_factory:
  name: "SVC"
```

**Default Parameters:**

| Param          | Description              | Default Value |
| -------------- | ------------------------ | ------------- |
| `kernel`       | Type of kernel           | `'rbf'`       |
| `C`            | Regularization parameter | `1.0`         |
| `gamma`        | Kernel coefficient       | `'scale'`     |
| `probability`  | Enable probability       | `True`        |
| `random_state` | Reproducibility          | `42`          |

---

### 🔹 `"GaussianNB"`

**Library**: `sklearn.naive_bayes.GaussianNB`  
**Run key**: `gaussian_nb_classifier_model`

```yaml
pipeline_factory:
  name: "GaussianNB"
```

**Default Parameters:**  
_No tunable parameters defined._

---

### 🔹 `"DecisionTree"`

**Library**: `sklearn.tree.DecisionTreeClassifier`  
**Run key**: `decision_tree_classifier_model`

```yaml
pipeline_factory:
  name: "DecisionTree"
```

**Default Parameters:**

| Param              | Description              | Default Value |
| ------------------ | ------------------------ | ------------- |
| `max_depth`        | Max depth of tree        | `10`          |
| `min_samples_leaf` | Minimum samples per leaf | `4`           |
| `criterion`        | Split quality metric     | `'gini'`      |
| `class_weight`     | Class balancing          | `'balanced'`  |
| `random_state`     | Reproducibility          | `42`          |

</details>


## 📈 Regression Models

<details>
<summary>click to expand section</summary>


### 🔹 `"LinearRegression"`

**Library**: `sklearn.linear_model.LinearRegression`  
**Run key**: `linear_regression_model`

```yaml
pipeline_factory:
  name: "LinearRegression"
```

**Default Parameters:**  
_No params required. Uses defaults._

---

### 🔹 `"RandomForestRegressor"`

**Library**: `sklearn.ensemble.RandomForestRegressor`  
**Run key**: `random_forest_regressor_model`

```yaml
pipeline_factory:
  name: "RandomForestRegressor"
```

**Default Parameters:**

| Param              | Description              | Default Value |
| ------------------ | ------------------------ | ------------- |
| `n_estimators`     | Number of trees          | `100`         |
| `max_depth`        | Max tree depth           | `10`          |
| `min_samples_leaf` | Minimum samples per leaf | `4`           |
| `random_state`     | Reproducibility          | `42`          |

---

### 🔹 `"XGBRegressor"`

**Library**: `xgboost.XGBRegressor`  
**Run key**: `xgboost_regressor_model`

```yaml
pipeline_factory:
  name: "XGBRegressor"
```

**Default Parameters:**

| Param           | Description               | Default Value        |
| --------------- | ------------------------- | -------------------- |
| `n_estimators`  | Number of boosting rounds | `100`                |
| `max_depth`     | Tree depth                | `5`                  |
| `learning_rate` | Step size shrinkage       | `0.1`                |
| `objective`     | Loss function             | `'reg:squarederror'` |
| `random_state`  | Reproducibility           | `42`                 |

---

### 🔹 `"DecisionTreeRegressor"`

**Library**: `sklearn.tree.DecisionTreeRegressor`  
**Run key**: `decision_tree_regressor_model`

```yaml
pipeline_factory:
  name: "DecisionTreeRegressor"
```

**Default Parameters:**

| Param              | Description       | Default Value |
| ------------------ | ----------------- | ------------- |
| `max_depth`        | Max tree depth    | `10`          |
| `min_samples_leaf` | Minimum leaf size | `4`           |
| `random_state`     | Reproducibility   | `42`          |

---

### 🔹 `"SVR"`

**Library**: `sklearn.svm.SVR`  
**Run key**: `svr_regressor_model`

```yaml
pipeline_factory:
  name: "SVR"
```

**Default Parameters:**

| Param    | Description             | Default Value |
| -------- | ----------------------- | ------------- |
| `kernel` | Kernel type             | `'rbf'`       |
| `C`      | Regularization strength | `1.0`         |
| `gamma`  | Kernel coefficient      | `'scale'`     |

</details>



### 📎 Quicklinks
> Return to the resource hub index: [Resource Hub Index](../hub_index.md)  
> Return to the top-level project overview: [Main Repository README](../../README.md)
