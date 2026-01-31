"""Unit tests for model factory."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from model_eval_suite.modeling.factory import pipeline_factory


class TestPipelineFactory:
    """Test pipeline factory functionality."""

    def test_create_logistic_regression_pipeline(self):
        """Test creation of logistic regression pipeline."""
        config = {
            "name": "LogisticRegression",
            "numeric_features": ["feature_1", "feature_2"],
            "categorical_features": ["cat_feature"],
            "params": {"random_state": 42},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "estimator" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["estimator"], LogisticRegression)

    def test_create_random_forest_pipeline(self):
        """Test creation of random forest pipeline."""
        config = {
            "name": "RandomForest",
            "numeric_features": ["feature_1"],
            "categorical_features": [],
            "params": {"n_estimators": 10, "random_state": 42},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["estimator"], RandomForestClassifier)

    def test_create_xgboost_pipeline(self):
        """Test creation of XGBoost pipeline."""
        config = {
            "name": "XGBoost",
            "numeric_features": ["feature_1", "feature_2"],
            "categorical_features": [],
            "params": {"n_estimators": 10, "random_state": 42},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["estimator"], XGBClassifier)

    def test_create_linear_regression_pipeline(self):
        """Test creation of linear regression pipeline."""
        config = {
            "name": "LinearRegression",
            "numeric_features": ["feature_1", "feature_2"],
            "categorical_features": ["cat_feature"],
            "params": {},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(pipeline.named_steps["estimator"], LinearRegression)

    def test_pipeline_with_no_categorical_features(self):
        """Test pipeline creation with only numeric features."""
        config = {
            "name": "LogisticRegression",
            "numeric_features": ["feature_1", "feature_2"],
            "categorical_features": [],
            "params": {"random_state": 42},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        # Should still have preprocessor for numeric features
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_with_no_numeric_features(self):
        """Test pipeline creation with only categorical features."""
        config = {
            "name": "LogisticRegression",
            "numeric_features": [],
            "categorical_features": ["cat_feature"],
            "params": {"random_state": 42},
        }

        pipeline = pipeline_factory(config)
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps

    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises appropriate error."""
        config = {
            "name": "InvalidModel",
            "numeric_features": ["feature_1"],
            "categorical_features": [],
            "params": {},
        }

        # Should raise an error or return None - check implementation
        with pytest.raises(ValueError):
            pipeline_factory(config)
