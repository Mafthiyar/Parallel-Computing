"""
data_loader.py
--------------
Handles CSV ingestion and initial data profiling.
"""

import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


def profile_data(df: pd.DataFrame) -> dict:
    """Return a summary profile of the dataset."""
    profile = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percent": (df.isnull().mean() * 100).round(2).to_dict(),
        "numeric_summary": df.describe().to_dict(),
        "duplicates": df.duplicated().sum()
    }
    logger.info(f"Data profile complete. Duplicates: {profile['duplicates']}")
    return profile


def save_csv(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")
