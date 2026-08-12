"""Training script for Bank Marketing model benchmark."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data import load_dataset, validate_dataset
from src.eda import run_eda
from src.evaluation import build_comparison_table, evaluate_model
from src.models import get_models
from src.preprocessing import prepare_data, transform_features
from src.utils import ensure_directories, save_joblib


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate Bank Marketing classifiers")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to semicolon-separated bank.csv file",
    )
    return parser.parse_args()


def train_models(data_path: Path) -> pd.DataFrame:
    """Execute full train/evaluate/persist pipeline."""
    ensure_directories([DATA_DIR, MODELS_DIR, REPORTS_DIR, ARTIFACTS_DIR])

    df = load_dataset(data_path)
    validate_dataset(df, TARGET_COLUMN)
    run_eda(df, TARGET_COLUMN, REPORTS_DIR)

    prepared = prepare_data(
        df=df,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    x_train_t, x_test_t = transform_features(
        preprocessor=prepared.preprocessor,
        x_train=prepared.x_train,
        x_test=prepared.x_test,
    )

    save_joblib(prepared.preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    save_joblib(prepared.target_encoder, ARTIFACTS_DIR / "target_encoder.joblib")
    save_joblib(prepared.x_train.columns.tolist(), ARTIFACTS_DIR / "feature_columns.joblib")

    results: list[dict] = []
    models = get_models(RANDOM_STATE)

    for model_name, model in models.items():
        model.fit(x_train_t, prepared.y_train)
        model_file_name = model_name.lower().replace(" ", "_") + ".joblib"
        compress = 3 if model_name == "Random Forest" else 0
        save_joblib(model, MODELS_DIR / model_file_name, compress=compress)

        metrics = evaluate_model(model, model_name, x_test_t, prepared.y_test)
        results.append(metrics)

    comparison = build_comparison_table(results)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    metrics_export = []
    for result in results:
        metrics_export.append(
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1": result["F1"],
                "ROC-AUC": result["ROC-AUC"],
                "MCC": result["MCC"],
                "Confusion Matrix": json.dumps(result["Confusion Matrix"]),
                "Classification Report": json.dumps(result["Classification Report"]),
            }
        )

    pd.DataFrame(metrics_export).to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    with (REPORTS_DIR / "detailed_model_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return comparison


def main() -> None:
    """Program entrypoint."""
    args = parse_args()
    comparison = train_models(args.data_path)
    print("Training complete.")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
