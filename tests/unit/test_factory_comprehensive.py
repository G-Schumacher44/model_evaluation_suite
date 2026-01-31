"""Comprehensive tests for model factory to increase coverage."""

import pytest
from sklearn.pipeline import Pipeline

from model_eval_suite.modeling.factory import pipeline_factory


class TestPipelineFactoryComprehensive:
    """Comprehensive tests for pipeline factory."""

    def test_all_classification_models(self, sample_classification_data):
        """Test all classification model types."""
        X = sample_classification_data.drop("target", axis=1)
        y = sample_classification_data["target"]

        models = [
            ("LogisticRegression", {}),
            ("RandomForest", {"n_estimators": 10, "random_state": 42}),
            ("XGBoost", {"n_estimators": 10, "random_state": 42}),
            ("SVC", {"kernel": "linear", "random_state": 42}),
            ("GaussianNB", {}),
            ("DecisionTree", {"max_depth": 3, "random_state": 42}),
        ]

        for model_name, params in models:
            config = {
                "name": model_name,
                "numeric_features": ["feature_1", "feature_2", "feature_3"],
                "categorical_features": ["cat_feature"],
                "params": params,
            }

            pipeline = pipeline_factory(config)
            assert isinstance(pipeline, Pipeline)

            # Try to fit
            try:
                pipeline.fit(X, y)
                predictions = pipeline.predict(X)
                assert len(predictions) == len(y)
            except Exception as e:
                pytest.fail(f"Failed to fit {model_name}: {e}")

    def test_all_regression_models(self, sample_regression_data):
        """Test all regression model types."""
        X = sample_regression_data.drop("target", axis=1)
        y = sample_regression_data["target"]

        models = [
            ("LinearRegression", {}),
            ("RandomForestRegressor", {"n_estimators": 10, "random_state": 42}),
            ("XGBRegressor", {"n_estimators": 10, "random_state": 42}),
            ("SVR", {"kernel": "linear"}),
            ("DecisionTreeRegressor", {"max_depth": 3, "random_state": 42}),
        ]

        for model_name, params in models:
            config = {
                "name": model_name,
                "numeric_features": ["feature_1", "feature_2"],
                "categorical_features": ["cat_feature"],
                "params": params,
            }

            pipeline = pipeline_factory(config)
            assert isinstance(pipeline, Pipeline)

            # Try to fit
            try:
                pipeline.fit(X, y)
                predictions = pipeline.predict(X)
                assert len(predictions) == len(y)
            except Exception as e:
                pytest.fail(f"Failed to fit {model_name}: {e}")

    def test_empty_params(self, sample_classification_data):
        """Test factory with empty params dict."""
        X = sample_classification_data.drop("target", axis=1)
        y = sample_classification_data["target"]

        config = {
            "name": "LogisticRegression",
            "numeric_features": ["feature_1"],
            "categorical_features": [],
            "params": {},
        }

        pipeline = pipeline_factory(config)
        pipeline.fit(X, y)
        assert pipeline is not None

    def test_only_one_feature_type(self, sample_classification_data):
        """Test with only numeric or only categorical features."""
        X = sample_classification_data.drop("target", axis=1)
        y = sample_classification_data["target"]

        # Only numeric
        config_numeric = {
            "name": "LogisticRegression",
            "numeric_features": ["feature_1", "feature_2"],
            "categorical_features": [],
            "params": {"random_state": 42},
        }

        pipeline = pipeline_factory(config_numeric)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

        # Only categorical
        config_cat = {
            "name": "LogisticRegression",
            "numeric_features": [],
            "categorical_features": ["cat_feature"],
            "params": {"random_state": 42},
        }

        pipeline = pipeline_factory(config_cat)
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_pipeline_has_preprocessor_step(self):
        """Test that pipeline always has preprocessor."""
        config = {
            "name": "LogisticRegression",
            "numeric_features": ["f1"],
            "categorical_features": [],
            "params": {},
        }

        pipeline = pipeline_factory(config)
        assert "preprocessor" in pipeline.named_steps
        assert "estimator" in pipeline.named_steps

    def test_pipeline_with_missing_features_in_config(self):
        """Test pipeline when feature lists are not specified."""
        config = {"name": "LogisticRegression", "params": {}}

        pipeline = pipeline_factory(config)
        # Should create pipeline with empty feature lists
        assert isinstance(pipeline, Pipeline)
