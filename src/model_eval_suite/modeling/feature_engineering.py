"""
⚙️ Optional feature engineering module for the evaluation suite.

This script defines a sample custom transformer (HREngineer) for demonstration purposes.
Users may override or disable feature engineering in their YAML config:

    feature_engineering:
        run: true       # or false to disable this step
        module: "model_eval_suite.modeling.feature_engineering"
        class_name: "HREngineer"

To integrate a custom transformer, provide its import path and class name in the override config.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HREngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for the Salifort dataset."""
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_transformed = X.copy()
        X_transformed['project_hours_interaction'] = X_transformed['number_project'] * X_transformed['average_montly_hours']
        return X_transformed