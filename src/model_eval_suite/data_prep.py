import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys
from model_eval_suite.utils.data_prep_config import load_data_prep_config

"""
📂 Data Preparation Utility

This script performs a one-time data split using a user-provided YAML config to define input paths and target columns. It assumes classification use and requires a valid target column for stratification.

Key Behavior:
- Loads raw data from disk
- Splits off a stratified holdout set (20%)
- Splits the remaining dev set into train/test (80/20 split)
- Saves results to `data/dev_data/` and `data/holdout_data/`

Run this once per dataset version to ensure consistent splits for modeling.

Usage:

CLI:
    $ python src/model_eval_suite/data_prep.py

Jupyter Notebook:
    from model_eval_suite.data_prep import main
    main()
"""

def main(config_path: str = "config/data_prep_config.yaml"):
    """Performs a one-time split of the raw data into train, test, and holdout sets."""

    config = load_data_prep_config(config_path)
    RAW_DATA_PATH = config.paths.input_data
    TRAIN_PATH = config.paths.train_data_path
    TEST_PATH = config.paths.test_data_path
    HOLDOUT_PATH = config.paths.holdout_data_path
    TARGET_COLUMN = config.target_column
    HOLDOUT_SIZE = 0.2
    TEST_SIZE_OF_DEV = 0.25  # 25% of dev split = 20% overall

    DEV_DIR = TRAIN_PATH.parent
    HOLDOUT_DIR = HOLDOUT_PATH.parent

    # --- ADDED: Safety Check ---
    if TRAIN_PATH.exists() or TEST_PATH.exists() or HOLDOUT_PATH.exists():
        print("⚠️ Output data files already exist. Skipping data split to avoid overwriting.")
        print(f" - Train: {TRAIN_PATH}")
        print(f" - Test: {TEST_PATH}")
        print(f" - Holdout: {HOLDOUT_PATH}")
        sys.exit(0) # Exit gracefully
    # -------------------------

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Split out holdout set (20%) from full dataset
    print(f"Performing initial holdout split...")
    dev_df, holdout_df = train_test_split(
        df, test_size=HOLDOUT_SIZE, random_state=42, stratify=df[TARGET_COLUMN]
    )

    # Split dev set into training (60%) and testing (20%) partitions
    print(f"Performing train/test split on development data...")
    train_df, test_df = train_test_split(
        dev_df, test_size=TEST_SIZE_OF_DEV, random_state=42, stratify=dev_df[TARGET_COLUMN]
    )

    # Ensure output directories exist
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save datasets to CSV files
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_PATH, index=False)
    
    print(f"✅ Train data saved to: {TRAIN_PATH} ({len(train_df)} rows)")
    print(f"✅ Test data saved to: {TEST_PATH} ({len(test_df)} rows)")
    print(f"✅ Holdout data saved to: {HOLDOUT_PATH} ({len(holdout_df)} rows)")

if __name__ == "__main__":
    main()