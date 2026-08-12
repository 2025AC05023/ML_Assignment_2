"""Model evaluation utilities for binary classification."""

from typing import Any

import pandas as pd
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


def _get_score(model: Any, x_test):
    """Get score vector for ROC-AUC depending on model capabilities."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def evaluate_model(model: Any, model_name: str, x_test, y_test) -> dict[str, Any]:
    """Compute assignment metrics for a single model."""
    y_pred = model.predict(x_test)
    y_score = _get_score(model, x_test)

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


def build_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create sorted model comparison table from evaluation results."""
    comparison = pd.DataFrame(
        [
            {
                "Model": result["Model"],
                "Accuracy": result["Accuracy"],
                "Precision": result["Precision"],
                "Recall": result["Recall"],
                "F1": result["F1"],
                "ROC-AUC": result["ROC-AUC"],
                "MCC": result["MCC"],
            }
            for result in results
        ]
    )

    comparison = comparison.sort_values(
        by=["ROC-AUC", "F1", "Accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    return comparison
