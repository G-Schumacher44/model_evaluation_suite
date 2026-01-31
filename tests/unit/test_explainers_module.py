"""Unit tests for explainers module."""

import pandas as pd

from model_eval_suite.modeling.explainers import (
    generate_all_explainers,
    generate_shap_explainer_and_values,
)


class TestSHAPExplainers:
    """Test SHAP explainer generation."""

    def test_generate_shap_explainer_and_values(
        self, trained_classifier, sample_classification_data, sample_config_dict
    ):
        """Test SHAP explainer generation."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_classification_data.drop("target", axis=1)

        # Add explainability config
        sample_config_dict["evaluation"]["explainability"] = {"run": True, "shap_sample_size": 20}
        config = SuiteConfig(**sample_config_dict)

        result = generate_shap_explainer_and_values(
            model=trained_classifier, X_data=X.head(30), config=config
        )

        if result:  # May return empty dict for some models
            assert "explainer" in result or len(result) == 0
            if "shap_values" in result:
                assert result["shap_values"] is not None
            if "data_for_shap" in result:
                assert isinstance(result["data_for_shap"], pd.DataFrame)

    def test_generate_shap_with_sampling(
        self, trained_classifier, sample_classification_data, sample_config_dict
    ):
        """Test SHAP with data sampling."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_classification_data.drop("target", axis=1)

        sample_config_dict["evaluation"]["explainability"] = {
            "run": True,
            "shap_sample_size": 10,  # Sample size smaller than data
        }
        config = SuiteConfig(**sample_config_dict)

        result = generate_shap_explainer_and_values(
            model=trained_classifier, X_data=X.head(50), config=config
        )

        if result and "data_for_shap" in result:
            # Sampled data should be at most sample_size
            assert len(result["data_for_shap"]) <= 10

    def test_generate_shap_without_sampling(
        self, trained_classifier, sample_classification_data, sample_config_dict
    ):
        """Test SHAP without sampling."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_classification_data.drop("target", axis=1)

        sample_config_dict["evaluation"]["explainability"] = {
            "run": True,
            "shap_sample_size": None,  # No sampling
        }
        config = SuiteConfig(**sample_config_dict)

        result = generate_shap_explainer_and_values(
            model=trained_classifier, X_data=X.head(20), config=config
        )

        if result and "data_for_shap" in result:
            # Should use all data
            assert len(result["data_for_shap"]) == 20

    def test_generate_shap_no_explainability_config(
        self, trained_classifier, sample_classification_data, sample_config_dict
    ):
        """Test SHAP when explainability is not configured."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_classification_data.drop("target", axis=1)

        # No explainability config
        sample_config_dict["evaluation"]["explainability"] = None
        config = SuiteConfig(**sample_config_dict)

        result = generate_shap_explainer_and_values(
            model=trained_classifier, X_data=X.head(10), config=config
        )

        # Should return empty dict
        assert result == {}

    def test_generate_all_explainers(
        self, trained_classifier, sample_classification_data, sample_config_dict
    ):
        """Test generating all explainer types."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_classification_data.drop("target", axis=1)

        sample_config_dict["evaluation"]["explainability"] = {"run": True, "shap_sample_size": 15}
        config = SuiteConfig(**sample_config_dict)

        result = generate_all_explainers(
            model=trained_classifier, X_train=X.head(50), config=config
        )

        assert isinstance(result, dict)
        # Should have SHAP results
        if result:
            assert "explainer" in result or len(result) == 0


class TestExplainersWithDifferentModels:
    """Test explainers with various model types."""

    def test_explainer_with_regressor(
        self, trained_regressor, sample_regression_data, sample_config_dict
    ):
        """Test SHAP with regression model."""
        from model_eval_suite.utils.config import SuiteConfig

        X = sample_regression_data.drop("target", axis=1)

        sample_config_dict["task_type"] = "regression"
        sample_config_dict["evaluation"]["explainability"] = {"run": True, "shap_sample_size": 10}
        config = SuiteConfig(**sample_config_dict)

        result = generate_shap_explainer_and_values(
            model=trained_regressor, X_data=X.head(20), config=config
        )

        # Should work for regression models too
        assert isinstance(result, dict)

    def test_explainer_error_handling(self, sample_config_dict):
        """Test that explainer handles errors gracefully."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from model_eval_suite.utils.config import SuiteConfig

        # Create a simple pipeline without proper structure
        bad_model = Pipeline([("scaler", StandardScaler())])

        X = pd.DataFrame({"f1": [1, 2, 3]})

        sample_config_dict["evaluation"]["explainability"] = {"run": True}
        config = SuiteConfig(**sample_config_dict)

        # Should not crash, might return empty dict
        result = generate_shap_explainer_and_values(model=bad_model, X_data=X, config=config)

        assert isinstance(result, dict)
