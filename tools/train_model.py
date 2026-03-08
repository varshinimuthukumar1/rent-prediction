from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_pipeline import PROCESSED_DATASET, build_dataset


DEFAULT_MODEL_PATH = Path("models/rent_price_model.joblib")

# Reusable feature set for training (pass as feature_cols=SELECTED_FEATURE_COLS)
SELECTED_FEATURE_COLS = [
    "livingSpace",
    "noRooms",
    "yearConstructed",
    "yearConstructedRange",
    "lastRefurbish",
    "floor",
    "numberOfFloors",
    "pricetrend",
    "thermalChar",
    "serviceCharge",
    "livingSpaceRange",
    "noRoomsRange",
    "avg_rent_per_sqm_plz",
    "avg_total_rent_regio3",
    "avg_total_rent_regio2",
]


def get_feature_target(
    df: pd.DataFrame,
    target_col: str = "totalRent",
    feature_cols: Optional[Sequence[str]] = None,
    drop_cols: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from the cleaned dataframe.

    If feature_cols is provided, use only those columns (minus target).
    Otherwise use all columns except default_drop and drop_cols.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].astype(float)

    if feature_cols is not None:
        # Use only the specified columns that exist and are not the target
        use_cols = [c for c in feature_cols if c in df.columns and c != target_col]
        X = df[use_cols].copy()
    else:
        # Columns that should not be used as predictors
        default_drop = {"scoutId", target_col, "rent_per_sqm"}
        if drop_cols:
            default_drop.update(drop_cols)
        feature_cols = [c for c in df.columns if c not in default_drop]
        X = df[feature_cols].copy()

    # Drop rows where target is missing
    target_mask = ~y.isna()
    X = X.loc[target_mask]
    y = y.loc[target_mask]

    # Drop feature columns that are entirely missing
    non_all_nan_cols = [c for c in X.columns if not X[c].isna().all()]
    X = X[non_all_nan_cols]

    return X, y


def build_preprocessing_and_model(X: pd.DataFrame, estimator=None) -> Pipeline:
    """
    Build a preprocessing + model pipeline for tabular data.
    """
    # Simple heuristic: treat object columns as categorical, rest as numeric
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "boolean"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    return pipeline


def get_estimator(model_name: str, model_params: Optional[dict] = None):
    """
    Build an estimator for the given model_name using the model factories.

    model_name: one of ["random_forest", "xgboost", "catboost", "lightgbm"]
    """
    if model_params is None:
        model_params = {}

    name = model_name.lower()

    if name == "random_forest":
        from models.random_forest import build_model

        return build_model(**model_params)

    if name == "xgboost":
        try:
            from models.xgboost import build_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "xgboost is not installed. Install it with 'pip install xgboost'."
            ) from exc
        return build_model(**model_params)

    if name == "catboost":
        try:
            from models.catboost import build_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "catboost is not installed. Install it with 'pip install catboost'."
            ) from exc
        return build_model(**model_params)

    if name == "lightgbm":
        try:
            from models.lightgbm import build_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lightgbm is not installed. Install it with 'pip install lightgbm'."
            ) from exc
        return build_model(**model_params)

    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        "Expected one of ['random_forest', 'xgboost', 'catboost', 'lightgbm']."
    )


def train_rent_model(
    processed_path: Optional[Path | str] = None,
    model_out: Optional[Path | str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int = 10000,
    model_name: str = "catboost",
    model_params: Optional[dict] = None,
    feature_cols: Optional[Sequence[str]] = None,
    exclude_cols: Optional[Sequence[str]] = None,
    run_name: Optional[str] = None,
) -> Path:
    """
    Train a rent price prediction model on the processed dataset.

    feature_cols: if set, use only these columns as features; else use all except target/scoutId/rent_per_sqm and exclude_cols.
    exclude_cols: extra columns to drop from features (ignored if feature_cols is set).
    run_name: if set, save model to models/rent_price_model_{run_name}.joblib (only when model_out is None).

    Returns the path where the trained model is saved.
    """
    if processed_path is None:
        # Ensure processed dataset exists
        if not PROCESSED_DATASET.exists():
            build_dataset()
        processed_path = PROCESSED_DATASET

    processed_path = Path(processed_path)
    df = pd.read_parquet(processed_path)

    # Downsample for faster training if dataset is large
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)

    X, y = get_feature_target(
        df,
        target_col="totalRent",
        feature_cols=feature_cols,
        drop_cols=exclude_cols,
    )

    # Build estimator (can be customized via model_name and model_params)
    estimator = get_estimator(model_name=model_name, model_params=model_params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_preprocessing_and_model(X_train, estimator=estimator)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:,.2f}")
    print(f"Test MAE : {mae:,.2f}")
    print(f"Test R^2 : {r2:,.3f}")

    # Save model
    if model_out is None:
        if run_name:
            model_out = Path("models") / f"rent_price_model_{run_name}.joblib"
        else:
            model_out = DEFAULT_MODEL_PATH
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_out)
    print(f"Saved trained model to: {model_out}")

    return model_out


if __name__ == "__main__":
    train_rent_model()

