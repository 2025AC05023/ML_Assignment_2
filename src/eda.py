"""EDA helpers to generate reusable data profiling artifacts."""

import json
from pathlib import Path

import pandas as pd


def run_eda(df: pd.DataFrame, target_column: str, reports_dir: Path) -> None:
    """Generate and save EDA artifacts for reproducible analysis."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_column,
        "target_distribution": df[target_column].astype(str).value_counts().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    with (reports_dir / "eda_overview.json").open("w", encoding="utf-8") as file:
        json.dump(overview, file, indent=2)

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_percentage": (df.isnull().mean().values * 100).round(3),
        }
    ).sort_values(by="missing_count", ascending=False)
    missing.to_csv(reports_dir / "missing_values_report.csv", index=False)

    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        numeric_df.describe().transpose().to_csv(reports_dir / "numeric_summary.csv")

    categorical_df = df.select_dtypes(exclude=["number"])
    if not categorical_df.empty:
        rows = []
        for column in categorical_df.columns:
            top_values = (
                categorical_df[column].astype(str).value_counts(dropna=False).head(5).to_dict()
            )
            rows.append(
                {
                    "column": column,
                    "unique_values": int(categorical_df[column].nunique(dropna=False)),
                    "top_values": json.dumps(top_values),
                }
            )
        pd.DataFrame(rows).to_csv(reports_dir / "categorical_summary.csv", index=False)
