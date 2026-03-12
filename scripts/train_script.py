"""
Run training across multiple models and hyperparameter configs; summarize results.
Usage: from project root with venv active: python scripts/train_script.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow importing tools and models from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
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

# Fixed split for reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ROWS = 10_000  # cap rows for speed

# (model_name, model_params, run_name) — add or edit as needed
RUNS = [
    # Random Forest
    ("random_forest", {"n_estimators": 50, "max_depth": 15}, "rf_50_15"),
    ("random_forest", {"n_estimators": 100, "max_depth": 20}, "rf_100_20"),
    # CatBoost
    ("catboost", {"n_estimators": 200, "depth": 6, "learning_rate": 0.1}, "cb_200_d6"),
    ("catboost", {"n_estimators": 500, "depth": 6, "learning_rate": 0.05}, "cb_500_d6"),
    # XGBoost (skip if not installed)
    ("xgboost", {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}, "xgb_200_d6"),
    ("xgboost", {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05}, "xgb_400_d6"),
    # LightGBM (skip if not installed)
    ("lightgbm", {"n_estimators": 200, "num_leaves": 31, "learning_rate": 0.1}, "lgb_200_31"),
    ("lightgbm", {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05}, "lgb_400_31"),
]


def main():
    # Ensure data exists
    if not PROCESSED_DATASET.exists():
        build_dataset()
    df = pd.read_parquet(PROCESSED_DATASET)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE)

    X, y = get_feature_target(
        df,
        target_col="totalRent",
        feature_cols=SELECTED_FEATURE_COLS,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for model_name, model_params, run_name in RUNS:
        try:
            est = get_estimator(model_name=model_name, model_params=model_params)
        except ModuleNotFoundError as e:
            print(f"Skipping {run_name} ({model_name}): {e}")
            continue
        pipeline = build_preprocessing_and_model(X_train, estimator=est)
        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        rmse_train = mean_squared_error(y_train, y_pred_train) ** 0.5
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_test = mean_squared_error(y_test, y_pred_test) ** 0.5
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        results.append(
            {
                "run_name": run_name,
                "model": model_name,
                "params": str(model_params),
                "train_rmse": round(rmse_train, 2),
                "train_mae": round(mae_train, 2),
                "train_r2": round(r2_train, 4),
                "test_rmse": round(rmse_test, 2),
                "test_mae": round(mae_test, 2),
                "test_r2": round(r2_test, 4),
            }
        )
        # Save this run's model
        out_dir = Path("models")
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, out_dir / f"rent_price_model_{run_name}.joblib")
        print(f"  {run_name}: train RMSE={rmse_train:.2f} MAE={mae_train:.2f} R²={r2_train:.4f}  |  test RMSE={rmse_test:.2f} MAE={mae_test:.2f} R²={r2_test:.4f}")

    if not results:
        print("No runs completed.")
        return

    summary = pd.DataFrame(results)
    summary = summary.sort_values("test_rmse").reset_index(drop=True)
    out_path = Path("reports/training_results.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print("\nSummary (sorted by test RMSE):")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
