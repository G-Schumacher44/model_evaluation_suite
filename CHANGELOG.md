# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.2.0] - 2026-01-31

### Changed
- **BREAKING:** MLflow tracking backend now defaults to SQLite (`sqlite:///mlflow.db`) instead of filesystem (`file:./mlruns`)
  - Eliminates MLflow deprecation warnings (filesystem backend deprecated Feb 2026)
  - Provides better performance and ACID transaction guarantees
  - See [MIGRATION.md](MIGRATION.md) for upgrade instructions
- Updated `.gitignore` to include `mlflow.db` and related SQLite files, `.claudeignore` for IDE optimization
- Improved error messages for SHAP force plot failures in dashboards (classification, regression, validation)
- Enhanced MLflow model logging with pickle security warning suppression for trusted models
- Updated all documentation headers in `resource_hub/` to v0.2.0 with new CI/Coverage badges

### Added

- **Modern Click-based CLI**: New `model-eval` command-line interface with 5 subcommands
  - `model-eval run` - Execute model evaluation pipelines
  - `model-eval init` - Generate configuration templates interactively
  - `model-eval prep` - Prepare and split datasets
  - `model-eval validate` - Validate champion models
  - `model-eval list-models` - Display available model types
- Rich terminal output with colored text, formatted tables, and progress indicators
- `environment.yml` for conda environment setup
- `mlflow_tracking_uri` configuration option in `SuiteConfig` with default `"sqlite:///mlflow.db"`
- Migration script (`scripts/migrate_mlflow_to_sqlite.py`) for users with existing filesystem data
- Comprehensive migration guide ([MIGRATION.md](MIGRATION.md))
- Updated MLflow documentation with SQLite best practices
- Completely rewritten CLI usage guide with command reference, examples, and troubleshooting
- `.claudeignore` file to optimize Claude Code IDE performance by excluding large data directories
- Warning suppression for verbose third-party library loggers (alembic, mlflow.store.db.utils)
- Warning suppression for SHAP JavaScript library messages in interactive notebooks
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing across Python 3.11/3.12 and Ubuntu/macOS
- **Test Suite**: 70 unit and integration tests with 28% code coverage (exceeds 25% threshold)
- Pytest configuration with parallel execution, coverage reporting, and HTML reports
- Test fixtures for classification/regression data and trained models
- Comprehensive tests for config loading, model factory, logging, feature engineering, and CLI
- New badges: CI status, Coverage, Python 3.11+, Tests passing

### Fixed
- MLflow filesystem backend deprecation warnings no longer appear by default
- Noisy alembic migration logs during MLflow initialization
- Confusing SHAP force plot error messages now show user-friendly guidance
- MLflow cloudpickle security warnings for trusted model serialization
- CI coverage collection issues resolved with global parallel test execution disabled

---

## [0.1.0] - 2024-08-06

### Added
- Initial release of Model Evaluation Suite
- YAML-driven configuration system with deep merge support
- Support for 11 scikit-learn models (6 classifiers, 5 regressors)
- Interactive Jupyter dashboards with SHAP explainability
- MLflow integration for experiment tracking and model registry
- Pre-model diagnostics (VIF, skewness, missing values)
- Automated audit alert system with configurable thresholds
- Champion model validation workflow
- HTML export for dashboards and reports
- Comprehensive documentation and resource hub

### Features
- **Classification Models:** LogisticRegression, RandomForest, XGBoost, SVC, GaussianNB, DecisionTree
- **Regression Models:** LinearRegression, RandomForestRegressor, XGBRegressor, SVR, DecisionTreeRegressor
- **Plots:** ROC/PR curves, confusion matrix, calibration, learning curves, SHAP summaries, residuals
- **Explainability:** SHAP force plots, feature importance, permutation importance
- **Validation:** Holdout evaluation, baseline comparison, segmentation analysis
- **Export:** Model serialization, HTML reports, CSV metrics logs

---

## Migration Notes

### Upgrading from v0.1.0 to v0.2.0

**Quick Start (No Data Preservation):**
```bash
mv mlruns mlruns_old_backup  # Optional backup
# Next run will create mlflow.db automatically
```

**Preserve Historical Data:**
```bash
pip install mlflow-export-import
python scripts/migrate_mlflow_to_sqlite.py
```

**Continue Using Filesystem (Not Recommended):**
```yaml
# In your config
mlflow_tracking_uri: "file:./mlruns"
```

See [MIGRATION.md](MIGRATION.md) for detailed instructions.

---

[Unreleased]: https://github.com/G-Schumacher44/model_evaluation_suite/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/G-Schumacher44/model_evaluation_suite/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/G-Schumacher44/model_evaluation_suite/releases/tag/v0.1.0
