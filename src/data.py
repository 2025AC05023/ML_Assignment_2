"""Dataset loading and validation routines."""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load the Bank Marketing dataset from a semicolon-separated CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path, sep=";")


def validate_dataset(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the dataframe has sufficient content and target column."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from dataset.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must include at least one feature and one target.")
