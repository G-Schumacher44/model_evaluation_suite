import importlib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, Any, Optional

# Model Imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

"""
🧠 Model Factory for evaluation suite.

This module handles creation and configuration of modeling pipelines.
- Selects appropriate estimator based on YAML config (e.g., LogisticRegression, RandomForest)
- Constructs a full scikit-learn pipeline with preprocessing and modeling
- Handles hyperparameter tuning using GridSearchCV if enabled
- Supports both classification and regression tasks via a unified interface

Primary Function:
- build_model_pipeline(): Returns a fitted pipeline ready for evaluation or prediction
"""

def pipeline_factory(factory_config: Dict[str, Any], fe_config: Optional[Dict[str, Any]] = None) -> Pipeline:
    """
    Builds a Scikit-learn pipeline from configuration dictionaries.
    """
    # Initialize list to hold pipeline components
    pipeline_steps = []

    # Dynamically load and attach a custom feature engineering transformer if specified
    if fe_config and fe_config.get('run', False):
        try:
            module = importlib.import_module(fe_config['module'])
            TransformerClass = getattr(module, fe_config['class_name'])
            pipeline_steps.append(('feature_engineering', TransformerClass()))
        except Exception as e:
            raise ValueError(f"Could not load feature engineering class. Error: {e}")
            
    # Retrieve feature lists for column-specific preprocessing
    numeric_features = factory_config.get('numeric_features', [])
    categorical_features = factory_config.get('categorical_features', [])
    
    # Define pipeline for numeric columns: variance filtering + scaling
    numeric_pipeline = make_pipeline(
        VarianceThreshold(),
        StandardScaler()
    )
    
    # Combine numeric and categorical preprocessing into a single transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    # Add preprocessing step to the pipeline
    pipeline_steps.append(('preprocessor', preprocessor))
    
    # Extract model hyperparameters from config
    params = factory_config.get('params', {})
    # Determine which model to use
    model_name = factory_config.get('name')

    # Add chosen estimator to the pipeline
    if model_name == 'RandomForest':
        pipeline_steps.append(('estimator', RandomForestClassifier(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'LogisticRegression':
        pipeline_steps.append(('estimator', LogisticRegression(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'XGBoost':
        pipeline_steps.append(('estimator', XGBClassifier(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'SVC':
        pipeline_steps.append(('estimator', SVC(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'GaussianNB':
        pipeline_steps.append(('estimator', GaussianNB(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'DecisionTree':
        pipeline_steps.append(('estimator', DecisionTreeClassifier(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'LinearRegression':
        pipeline_steps.append(('estimator', LinearRegression(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'RandomForestRegressor':
        pipeline_steps.append(('estimator', RandomForestRegressor(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'XGBRegressor':
        pipeline_steps.append(('estimator', XGBRegressor(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'DecisionTreeRegressor':
        pipeline_steps.append(('estimator', DecisionTreeRegressor(**params)))
    # Add chosen estimator to the pipeline
    elif model_name == 'SVR':
        pipeline_steps.append(('estimator', SVR(**params)))
    else:
        raise ValueError(f"Unknown pipeline factory name: {model_name}")

    # Construct and return full pipeline
    return Pipeline(pipeline_steps)