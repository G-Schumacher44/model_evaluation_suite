from pathlib import Path
from pydantic import BaseModel

class DataPrepPaths(BaseModel):
    input_data: Path
    train_data_path: Path
    test_data_path: Path
    holdout_data_path: Path

class DataPrepConfig(BaseModel):
    paths: DataPrepPaths
    target_column: str

import yaml

def load_data_prep_config(path: str) -> DataPrepConfig:
    """
    Loads a YAML config and parses it into a DataPrepConfig object.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Flatten modeling.target_column if nested
    if "modeling" in raw and "target_column" in raw["modeling"]:
        raw["target_column"] = raw["modeling"]["target_column"]

    return DataPrepConfig(**raw)