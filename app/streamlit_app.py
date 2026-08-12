"""Streamlit dashboard for Bank Marketing model scoring and analysis."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, TARGET_COLUMN
from src.data import validate_dataset
from src.utils import load_joblib

st.set_page_config(page_title="Bank Marketing ML Dashboard", layout="wide")


@st.cache_resource
def load_artifacts() -> dict:
    """Load trained models and preprocessing artifacts."""
    preprocessor = load_joblib(ARTIFACTS_DIR / "preprocessor.joblib")
    target_encoder = load_joblib(ARTIFACTS_DIR / "target_encoder.joblib")
    feature_columns = load_joblib(ARTIFACTS_DIR / "feature_columns.joblib")

    model_paths = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Decision Tree": MODELS_DIR / "decision_tree.joblib",
        "KNN": MODELS_DIR / "knn.joblib",
        "Gaussian Naive Bayes": MODELS_DIR / "gaussian_naive_bayes.joblib",
        "Random Forest": MODELS_DIR / "random_forest.joblib",
    }

    models = {name: load_joblib(path) for name, path in model_paths.items()}
    return {
        "preprocessor": preprocessor,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "models": models,
    }


@st.cache_data
def load_dataframe(path):
    """Read CSV from path if available."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_model_results() -> dict:
    """Load saved model evaluation results from training reports."""
    results_path = REPORTS_DIR / "detailed_model_results.json"
    if not results_path.exists():
        return {}

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    return {result["Model"]: result for result in results}


def render_confusion_matrix(matrix):
    """Render confusion matrix heatmap."""
    fig, axis = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    st.pyplot(fig)


def render_selected_model_evaluation(model_result: dict) -> None:
    """Render saved evaluation metrics for the selected model."""
    st.subheader("Selected Model Evaluation")

    if not model_result:
        st.warning("No saved evaluation results found for the selected model. Run training first.")
        return

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC"]
    metric_labels = {"F1": "F1 Score"}
    metric_cols = st.columns(3)
    for index, metric_name in enumerate(metric_names):
        metric_cols[index % 3].metric(
            metric_labels.get(metric_name, metric_name),
            f"{model_result[metric_name]:.4f}",
        )

    render_confusion_matrix(model_result["Confusion Matrix"])

    st.write("Classification Report")
    report = pd.DataFrame(model_result["Classification Report"]).transpose()
    st.dataframe(report, use_container_width=True)


def main() -> None:
    """Render full Streamlit application."""
    st.title("Bank Marketing Classification Workbench")
    st.caption("Upload bank.csv, select a model, inspect predictions, and compare performance.")

    with st.sidebar:
        st.header("Model Selection")
        selected_model_name = st.selectbox(
            "Choose classifier",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Gaussian Naive Bayes",
                "Random Forest",
            ],
        )

    artifacts = load_artifacts()
    model = artifacts["models"][selected_model_name]
    preprocessor = artifacts["preprocessor"]
    target_encoder = artifacts["target_encoder"]
    feature_columns = artifacts["feature_columns"]

    st.subheader("Training EDA Reports")
    missing_report = load_dataframe(REPORTS_DIR / "missing_values_report.csv")
    comparison_report = load_dataframe(REPORTS_DIR / "model_comparison.csv")
    model_results = load_model_results()

    overview_path = REPORTS_DIR / "eda_overview.json"
    if overview_path.exists():
        with overview_path.open("r", encoding="utf-8") as file:
            overview = json.load(file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", overview["rows"])
        col2.metric("Columns", overview["columns"])
        col3.metric("Target", overview["target_column"])

        st.write("Target distribution")
        target_df = pd.DataFrame.from_dict(
            overview["target_distribution"],
            orient="index",
            columns=["count"],
        )
        st.dataframe(target_df, use_container_width=True)

    if not missing_report.empty:
        st.write("Missing values summary")
        st.dataframe(missing_report, use_container_width=True)

    uploaded_file = st.file_uploader("Upload semicolon-separated bank.csv", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, sep=";")
        validate_dataset(data, TARGET_COLUMN)

        st.subheader("Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)

        y_actual = data[TARGET_COLUMN].astype(str)
        features = data.drop(columns=[TARGET_COLUMN])

        missing_required = [col for col in feature_columns if col not in features.columns]
        if missing_required:
            st.error("Uploaded dataset is missing required columns: " + ", ".join(missing_required))
            st.stop()

        features = features[feature_columns]
        x_transformed = preprocessor.transform(features)

        y_pred_encoded = model.predict(x_transformed)
        y_pred_labels = target_encoder.inverse_transform(y_pred_encoded.astype(int))
        y_actual_encoded = target_encoder.transform(y_actual)

        prediction_df = features.copy()
        prediction_df["actual"] = y_actual.values
        prediction_df["prediction"] = y_pred_labels

        st.subheader("Predictions")
        st.dataframe(prediction_df.head(25), use_container_width=True)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_transformed)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(x_transformed)
        else:
            y_score = y_pred_encoded

        metrics = {
            "Accuracy": accuracy_score(y_actual_encoded, y_pred_encoded),
            "Precision": precision_score(y_actual_encoded, y_pred_encoded, zero_division=0),
            "Recall": recall_score(y_actual_encoded, y_pred_encoded, zero_division=0),
            "F1": f1_score(y_actual_encoded, y_pred_encoded, zero_division=0),
            "ROC-AUC": roc_auc_score(y_actual_encoded, y_score),
            "MCC": matthews_corrcoef(y_actual_encoded, y_pred_encoded),
        }

        st.subheader("Metrics")
        metric_cols = st.columns(3)
        items = list(metrics.items())
        for index, (name, value) in enumerate(items):
            metric_cols[index % 3].metric(name, f"{value:.4f}")

        cm = confusion_matrix(y_actual_encoded, y_pred_encoded)
        render_confusion_matrix(cm)

        st.subheader("Classification Report")
        report = classification_report(
            y_actual_encoded,
            y_pred_encoded,
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

        st.download_button(
            label="Download Predictions CSV",
            data=prediction_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

    render_selected_model_evaluation(model_results.get(selected_model_name, {}))

    st.subheader("Model Comparison")
    if comparison_report.empty:
        st.warning("No model comparison file found. Run training first.")
    else:
        st.dataframe(comparison_report, use_container_width=True)


if __name__ == "__main__":
    main()
