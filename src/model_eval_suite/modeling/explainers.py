import shap
import pandas as pd
from typing import Any, Dict
from model_eval_suite.utils.config import SuiteConfig

"""
🔍 Explainers module for model evaluation.

This module manages SHAP-based explainability:
- Generates SHAP explainer objects using the best method for the model type
- Transforms raw input data using the pipeline for accurate attribution
- Samples data for SHAP value generation if configured
- Returns explainer objects, SHAP values, and associated input data

Used during model evaluation to support local and global interpretation plots.
"""

def generate_shap_explainer_and_values(model: Any, X_data: pd.DataFrame, config: SuiteConfig) -> Dict:
    """Creates a SHAP explainer, calculates values, and prepares data for reuse."""
    try:
        # Extract the preprocessing steps from the pipeline (all but final estimator)
        transformer_pipeline = model[:-1]
        # Extract the final model (e.g., classifier)
        classifier = model[-1]
        # Apply preprocessing to input data
        X_transformed = transformer_pipeline.transform(X_data)
        # Retrieve named preprocessor to extract feature names
        preprocessor = transformer_pipeline.named_steps['preprocessor']
        final_features = preprocessor.get_feature_names_out()
        # Build DataFrame from transformed data with original indexing and feature names
        X_transformed_df = pd.DataFrame(X_transformed, columns=final_features, index=X_data.index)

        # Initialize SHAP explainer using final classifier and transformed inputs
        explainer = shap.Explainer(classifier, X_transformed_df)
        
        # Load explainability-related config from evaluation section
        expl_cfg = config.evaluation.explainability
        if not expl_cfg:
            return {}
            
        # Determine if sampling is enabled for SHAP generation
        sample_size = expl_cfg.get('shap_sample_size')

        # Optionally sample a subset of data for efficiency
        if sample_size and sample_size < len(X_transformed_df):
            data_for_shap = X_transformed_df.sample(n=sample_size, random_state=42)
        else:
            data_for_shap = X_transformed_df
            
        # Compute SHAP values for the (sampled or full) transformed dataset
        shap_values = explainer(data_for_shap)
        
        # Return all SHAP artifacts needed downstream
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "data_for_shap": data_for_shap
        }
    except Exception as e:
        print(f"⚠️ SHAP explainer creation failed: {e}")
        return {}

def generate_all_explainers(model: Any, X_train: pd.DataFrame, config: SuiteConfig) -> Dict:
    """Orchestrates generation of all configured explainability objects."""
    explainer_results = {}
    expl_cfg = config.evaluation.explainability

    if expl_cfg and expl_cfg.get('run_shap', False):
        shap_data = generate_shap_explainer_and_values(model, X_train, config)
        if shap_data:
            explainer_results.update(shap_data)
            
    return explainer_results