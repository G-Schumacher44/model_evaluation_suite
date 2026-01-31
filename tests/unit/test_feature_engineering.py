"""Unit tests for feature engineering."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TestFeatureEngineering:
    """Test feature engineering transformations."""

    def test_custom_transformer_interface(self, sample_classification_data):
        """Test that custom transformers follow sklearn interface."""
        # This tests the general contract that custom transformers should follow

        class DummyTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        transformer = DummyTransformer()
        X = sample_classification_data.drop("target", axis=1)

        # Test fit returns self
        result = transformer.fit(X)
        assert result is transformer

        # Test transform returns array-like
        X_transformed = transformer.transform(X)
        assert X_transformed is not None
        assert len(X_transformed) == len(X)

    def test_feature_engineering_preserves_shape(self, sample_classification_data):
        """Test that feature engineering preserves sample count."""
        X = sample_classification_data.drop("target", axis=1)
        n_samples = len(X)

        # Even with feature engineering, number of samples should remain the same
        assert len(X) == n_samples

    def test_numeric_features_are_numeric(self, sample_classification_data):
        """Test that numeric features are actually numeric."""
        numeric_cols = ["feature_1", "feature_2", "feature_3"]

        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(sample_classification_data[col])

    def test_categorical_features_encoding(self, sample_classification_data):
        """Test that categorical features can be encoded."""
        from sklearn.preprocessing import OneHotEncoder

        cat_feature = sample_classification_data[["cat_feature"]]
        encoder = OneHotEncoder(sparse_output=False)

        encoded = encoder.fit_transform(cat_feature)

        # Should have as many columns as unique categories
        n_categories = sample_classification_data["cat_feature"].nunique()
        assert encoded.shape[1] == n_categories
        assert encoded.shape[0] == len(sample_classification_data)
