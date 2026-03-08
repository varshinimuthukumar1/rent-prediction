"""
Train with different feature sets and compare metrics to judge which works best.
Usage: from project root with venv active: python scripts/train_feature_sets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tools.train_model import (
    PROCESSED_DATASET,
    SELECTED_FEATURE_COLS,
    build_dataset,
    build_preprocessing_and_model,
    get_estimator,
    get_feature_target,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ROWS = 10_000
MODEL_NAME = "catboost"  # single model to compare feature sets fairly

# (run_name, feature_cols or None for "all columns except default drops")
FEATURE_SETS = [
    ("selected", SELECTED_FEATURE_COLS),
    ("minimal", ["livingSpace", "noRooms", "avg_rent_per_sqm_plz", "serviceCharge"]),
    ("selected_no_plz", [c for c in SELECTED_FEATURE_COLS if c != "avg_rent_per_sqm_plz"]),
    ("core_numeric", [
        "livingSpace", "noRooms", "yearConstructed", "floor", "numberOfFloors",
        "pricetrend", "thermalChar", "serviceCharge", "avg_rent_per_sqm_plz",
    ]),
    ("with_location", SELECTED_FEATURE_COLS + ["regio1", "typeOfFlat"]),
    ("all", None),  # None = use all columns (except scoutId, totalRent, rent_per_sqm)
]


def main():
    if not PROCESSED_DATASET.exists():
        build_dataset()
    df = pd.read_parquet(PROCESSED_DATASET)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

    results = []

    for run_name, feature_cols in FEATURE_SETS:
        X, y = get_feature_target(
            df,
            target_col="totalRent",
            feature_cols=feature_cols,
        )
        n_features = X.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        pipeline = build_preprocessing_and_model(X_train, estimator=get_estimator(MODEL_NAME))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "feature_set": run_name,
            "n_features": n_features,
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 4),
        })
        print(f"  {run_name} (n={n_features}): RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

    summary = pd.DataFrame(results)
    summary = summary.sort_values("rmse").reset_index(drop=True)
    out_path = Path("reports/feature_set_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print("\nSummary (sorted by RMSE, best first):")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
