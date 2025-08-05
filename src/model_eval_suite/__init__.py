

"""
Model Evaluation Suite
A toolkit for running reproducible model evaluation experiments.
"""

# Expose the primary user-facing functions at the top level of the package
from .run_pipeline import main as run_experiment
from .validate_champion import validate_and_display as validate_champion
from .data_prep import main as prep_data

__version__ = "0.1.0"