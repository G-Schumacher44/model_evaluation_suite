# Test Suite

This directory contains the test suite for the Model Evaluation Suite project.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_config.py       # Configuration loading tests
│   ├── test_factory.py      # Model factory tests
│   ├── test_metrics.py      # Metrics calculation tests
│   ├── test_explainers.py   # SHAP explainer tests
│   ├── test_logging.py      # Logging configuration tests
│   └── test_feature_engineering.py
├── integration/             # Integration tests for workflows
│   └── test_pipeline_integration.py
└── fixtures/                # Test data and fixtures
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=src/model_eval_suite --cov-report=html
```

### Run specific test file
```bash
pytest tests/unit/test_config.py
```

### Run specific test class or function
```bash
pytest tests/unit/test_config.py::TestConfigLoading::test_load_config_from_dict
```

### Run tests in parallel
```bash
pytest -n auto
```

### Run with verbose output
```bash
pytest -v
```

## Coverage Requirements

- **Current coverage: 25%** (51 passing tests)
- Minimum threshold: 20%
- Target coverage: 50%+
- Critical modules coverage:
  - `utils/config.py`: 100%
  - `utils/logging.py`: 100%
  - `modeling/factory.py`: 89%
  - `modeling/explainers.py`: 85%
  - `classification/class_metrics.py`: 82%

## Writing Tests

### Fixtures

Use shared fixtures from `conftest.py`:
- `sample_classification_data` - Sample binary classification dataset
- `sample_regression_data` - Sample regression dataset
- `trained_classifier` - Pre-trained classifier pipeline
- `trained_regressor` - Pre-trained regressor pipeline
- `temp_dir` - Temporary directory for test files
- `sample_config_dict` - Valid configuration dictionary

### Example Test

```python
import pytest

def test_my_function(sample_classification_data, temp_dir):
    """Test description."""
    # Arrange
    X = sample_classification_data.drop('target', axis=1)

    # Act
    result = my_function(X)

    # Assert
    assert result is not None
    assert len(result) == len(X)
```

## CI/CD

Tests run automatically on:
- Push to main/develop branches
- Pull requests to main/develop
- Multiple Python versions (3.11, 3.12)
- Multiple OS (Ubuntu, macOS)

See `.github/workflows/ci.yml` for CI configuration.

## Coverage Reports

After running tests with coverage, open the HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```
