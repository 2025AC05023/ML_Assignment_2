"""Feature preprocessing and dataset splitting utilities."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    """Container for prepared data and preprocessing artifacts."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    target_encoder: LabelEncoder


def build_preprocessor(x_frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    numeric_features = x_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> PreparedData:
    """Split data, encode target, and prepare preprocessing pipeline."""
    x_frame = df.drop(columns=[target_column])
    y_series = df[target_column].astype(str)

    target_encoder = LabelEncoder()
    y_encoded = pd.Series(target_encoder.fit_transform(y_series), index=y_series.index)

    x_train, x_test, y_train, y_test = train_test_split(
        x_frame,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    preprocessor = build_preprocessor(x_train)

    return PreparedData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        target_encoder=target_encoder,
    )


def transform_features(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
):
    """Fit on train and transform train/test feature matrices."""
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    return x_train_t, x_test_t
