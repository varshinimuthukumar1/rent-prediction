"""
Train with different models and feature sets; compare metrics in one table.
Set MODELS to loop over estimators (e.g. catboost, xgboost, lightgbm, random_forest).
Set FEATURE_SETS to loop over (run_name, feature_cols). Results include both dimensions.
Usage: from project root with venv active: python scripts/train_feature_sets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tools.train_model import (
    PROCESSED_DATASET,
    SELECTED_FEATURE_COLS,
    TEXT_DERIVED_FEATURE_COLS,
    build_dataset,
    build_preprocessing_and_model,
    get_estimator,
    get_feature_target,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ROWS = 10_000

# Models to loop over (with feature sets). Use one name for single-model comparison.
MODELS = ["catboost","xgboost","lightgbm","random_forest"]

# Use cleaned dataset so target is log(totalRent) and engineered features exist.
# build_dataset() reads from raw/merged and writes PROCESSED_DATASET (cleaned + LLM cols if present).

# (run_name, feature_cols or None for "all columns except default drops")
# Only include text_derived set when TEXT_DERIVED_FEATURE_COLS is non-empty (otherwise it duplicates "selected")
_feature_sets_base = [
    ("selected", SELECTED_FEATURE_COLS),
    ("minimal", ["livingSpace", "noRooms", "avg_rent_per_sqm_plz", "serviceCharge"]),
    ("selected_no_plz", [c for c in SELECTED_FEATURE_COLS if c != "avg_rent_per_sqm_plz"]),
    ("core_numeric", [
        "livingSpace", "noRooms", "yearConstructed", "floor", "numberOfFloors",
        "pricetrend", "thermalChar", "serviceCharge", "avg_rent_per_sqm_plz",
    ]),
]
FEATURE_SETS = _feature_sets_base + (
    [("text_derived", SELECTED_FEATURE_COLS + TEXT_DERIVED_FEATURE_COLS)] if TEXT_DERIVED_FEATURE_COLS else []
)


def main():
    # So edits to tools.train_model (e.g. TEXT_DERIVED_FEATURE_COLS) are always picked up
    import importlib
    import tools.train_model as _tm
    importlib.reload(_tm)
    TEXT_DERIVED = getattr(_tm, "TEXT_DERIVED_FEATURE_COLS", [])
    FEATURE_SETS_RUNTIME = _feature_sets_base + (
        [("text_derived", _tm.SELECTED_FEATURE_COLS + TEXT_DERIVED)] if TEXT_DERIVED else []
    )

    # Always use cleaned dataset: log(totalRent), engineered features, and LLM columns when available
    if not PROCESSED_DATASET.exists():
        build_dataset()
    df = pd.read_parquet(PROCESSED_DATASET)
    requested = {c for _, cols in FEATURE_SETS_RUNTIME for c in (cols or [])}
    missing_text = requested & set(TEXT_DERIVED) - set(df.columns)
    if missing_text:
        print(f"Note: text-derived columns missing in data (e.g. {list(missing_text)[:3]}...); those feature sets will use fewer columns.")
    print(f"TEXT_DERIVED_FEATURE_COLS: {len(TEXT_DERIVED)} columns → {'including' if TEXT_DERIVED else 'skipping'} 'text_derived' feature set")
    print(f"Using cleaned dataset: {PROCESSED_DATASET} (target=log(totalRent), {len(df)} rows)")
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

    results = []

    for model_name in MODELS:
        for run_name, feature_cols in FEATURE_SETS_RUNTIME:
            try:
                estimator = get_estimator(model_name)
            except ModuleNotFoundError as e:
                print(f"Skipping {model_name} / {run_name}: {e}")
                continue
            X, y = get_feature_target(
                df,
                target_col="totalRent",
                feature_cols=feature_cols,
            )
            n_features = X.shape[1]
            if n_features == 0:
                print(f"  Warning: {run_name} has 0 features (requested columns missing in dataset) -> constant predictions, same metrics.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            pipeline = build_preprocessing_and_model(X_train, estimator=estimator)
            pipeline.fit(X_train, y_train)
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            rmse_train = mean_squared_error(y_train, y_pred_train) ** 0.5
            mae_train = mean_absolute_error(y_train, y_pred_train)
            r2_train = r2_score(y_train, y_pred_train)
            rmse_test = mean_squared_error(y_test, y_pred_test) ** 0.5
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)
            # Compute metrics in original euro scale (antilog of totalRent)
            y_train_eur = np.exp(y_train)
            y_test_eur = np.exp(y_test)
            y_pred_train_eur = np.exp(y_pred_train)
            y_pred_test_eur = np.exp(y_pred_test)
            rmse_train_eur = mean_squared_error(y_train_eur, y_pred_train_eur) ** 0.5
            mae_train_eur = mean_absolute_error(y_train_eur, y_pred_train_eur)
            rmse_test_eur = mean_squared_error(y_test_eur, y_pred_test_eur) ** 0.5
            mae_test_eur = mean_absolute_error(y_test_eur, y_pred_test_eur)
            results.append({
                "model": model_name,
                "feature_set": run_name,
                "n_features": n_features,
                "train_rmse": round(rmse_train, 2),
                "train_mae": round(mae_train, 2),
                "train_r2": round(r2_train, 4),
                "test_rmse": round(rmse_test, 2),
                "test_mae": round(mae_test, 2),
                "test_r2": round(r2_test, 4),
                "train_rmse_eur": round(rmse_train_eur, 1),
                "train_mae_eur": round(mae_train_eur, 1),
                "test_rmse_eur": round(rmse_test_eur, 1),
                "test_mae_eur": round(mae_test_eur, 1),
            })
            print(
                f"  [{model_name}] {run_name} (n={n_features}): "
                f"train RMSE={rmse_train:.2f} (≈{rmse_train_eur:,.0f} €) "
                f"MAE={mae_train:.2f} (≈{mae_train_eur:,.0f} €) R²={r2_train:.4f}  |  "
                f"test RMSE={rmse_test:.2f} (≈{rmse_test_eur:,.0f} €) "
                f"MAE={mae_test:.2f} (≈{mae_test_eur:,.0f} €) R²={r2_test:.4f}"
            )

    summary = pd.DataFrame(results)
    summary = summary.sort_values("test_rmse").reset_index(drop=True)
    out_path = Path("reports/feature_set_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print("\nSummary (sorted by test RMSE, best first):")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
