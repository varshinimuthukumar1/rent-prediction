from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RAW_DIR = Path("data/apartment-rental-offers-in-germany")
RAW_PARQUET = Path("data/processed/immo_merged_with_llm_features.parquet")
RAW_CSV = RAW_DIR / "immo_data.csv"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DATASET = Path("data/processed/immo_merged_with_llm_features_cleaned.parquet")

REPORTS_DIR = Path("reports/eda")

# Columns we do not want to use as model features
UNUSED_FEATURES = [
    "description",
    "facilities",
]


def load_raw_dataset(path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load the raw ImmoScout dataset from Parquet (preferred) or CSV.
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV)

    raise FileNotFoundError(
        f"Could not find raw data. Expected at {RAW_PARQUET} or {RAW_CSV}."
    )


def clean_immo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning and filtering to get a modelling-ready dataset.

    - Drop duplicate listings by scoutId (if present)
    - Parse dates
    - Convert obvious boolean columns
    - Ensure numeric types for key numeric columns
    - Filter out extreme / invalid values
    - Drop rows missing key fields
    - Create target column 'rent_per_sqm'
    """
    df = df.copy()

    # Drop duplicate listings
    if "scoutId" in df.columns:
        df = df.drop_duplicates(subset="scoutId")

    # Parse date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert TRUE/FALSE (or True/False) strings, or already-boolean, to booleans where appropriate
    # (luxury_score is ordinal 1–10 from LLM, so not in this list)
    # Parquet can load as Python bool; CSV/LLM often give "True"/"False" or "TRUE"/"FALSE"
    bool_candidates = ["newlyConst", "balcony", "hasKitchen", "cellar", "lift", "garden", "floor_heating", "guest_toilet", "built_in_kitchen", "garage_available", "dishwasher", "bathtub", "parquet_floor","green_view", "quiet_neighborhood", "near_public_transport"]
    bool_map = {True: True, False: False, "TRUE": True, "FALSE": False, "True": True, "False": False}
    for col in bool_candidates:
        if col in df.columns:
            df[col] = (
                df[col]
                .map(bool_map)
                .astype("boolean")
            )
            # Treat missing / empty as False (e.g. "not mentioned" = not present)
            df[col] = df[col].fillna(False)

    # Ensure numeric types for key numeric features (including ordinal 1–10 style columns)
    numeric_cols = [
        "serviceCharge", "totalRent", "baseRent", "livingSpace", "noRooms",
        "luxury_score",  # LLM ordinal 1–10; keep numeric for modelling
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter out obviously invalid or extreme values
    mask = pd.Series(True, index=df.index)

    if "livingSpace" in df.columns:
        mask &= df["livingSpace"].between(10, 400)  # m²
    if "totalRent" in df.columns:
        mask &= df["totalRent"].between(100, 10000)  # €
    if "noRooms" in df.columns:
        mask &= df["noRooms"].between(1, 10)

    df = df[mask]

    # Drop rows missing key fields required for modelling
    key_columns = ["totalRent", "livingSpace", "regio1", "typeOfFlat"]
    existing_keys = [c for c in key_columns if c in df.columns]
    if existing_keys:
        df = df.dropna(subset=existing_keys)

    # Create target variable: rent per square meter (using raw totalRent)
    if {"totalRent", "livingSpace"}.issubset(df.columns):
        df["rent_per_sqm"] = df["totalRent"] / df["livingSpace"]

    # Replace totalRent with log(totalRent) for modelling
    if "totalRent" in df.columns:
        df["totalRent"] = np.log(df["totalRent"])

    # Average rent per sqm per PLZ (new feature)
    if {"geo_plz", "rent_per_sqm"}.issubset(df.columns):
        df["avg_rent_per_sqm_plz"] = df.groupby("geo_plz")["rent_per_sqm"].transform("mean")

    # Average total rent per regio3 (new feature), leave-one-out to avoid target leakage
    if {"regio3", "totalRent"}.issubset(df.columns):
        g = df.groupby("regio3")["totalRent"]
        n = g.transform("count")
        leave_one_out_sum = g.transform("sum") - df["totalRent"]
        leave_one_out_count = n - 1
        avg_other = leave_one_out_sum / leave_one_out_count.replace(0, 1)
        global_mean = df["totalRent"].mean()
        # Use mean of other rows in same regio3; for single-row regio3 use overall mean
        df["avg_total_rent_regio3"] = avg_other.where(n > 1, global_mean).fillna(global_mean)

    # Average total rent per regio2 (new feature), leave-one-out to avoid target leakage
    if {"regio2", "totalRent"}.issubset(df.columns):
        g = df.groupby("regio2")["totalRent"]
        n = g.transform("count")
        leave_one_out_sum = g.transform("sum") - df["totalRent"]
        leave_one_out_count = n - 1
        avg_other = leave_one_out_sum / leave_one_out_count.replace(0, 1)
        global_mean = df["totalRent"].mean()
        df["avg_total_rent_regio2"] = avg_other.where(n > 1, global_mean).fillna(global_mean)

    # Drop text-heavy or unwanted columns that we do not want as features
    cols_to_drop = [c for c in UNUSED_FEATURES if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def build_dataset(
    raw_path: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
) -> Path:
    """
    Full pipeline: load raw data, clean it, and persist as Parquet.
    """
    df_raw = load_raw_dataset(raw_path)
    df_clean = clean_immo_data(df_raw)

    if out_path is None:
        out_path = PROCESSED_DATASET
    out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_path, index=False)

    return out_path


def run_basic_eda(df: pd.DataFrame, out_dir: Optional[Path | str] = None) -> None:
    """
    Create basic exploratory plots to understand feature relationships.

    - Correlation heatmap for numeric features
    - Pairplot for a small subset of key numeric features
    - Mean rent_per_sqm by region (top 15)

    Plots are saved under reports/eda by default.
    """
    if out_dir is None:
        out_dir = REPORTS_DIR
    # Create a unique subfolder per run so previous plots are preserved
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_dir) / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Correlation heatmap for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation heatmap (numeric features)")
        plt.tight_layout()
        plt.savefig(out_dir / "correlation_heatmap.png")
        plt.close()

    # Pairplot for a subset of important numeric features
    pairplot_cols = [
        col
        for col in ["livingSpace", "totalRent", "rent_per_sqm", "noRooms", "serviceCharge"]
        if col in df.columns
    ]
    if len(pairplot_cols) >= 2:
        sample_df = df[pairplot_cols]
        if len(sample_df) > 1000:
            sample_df = sample_df.sample(n=1000, random_state=42)
        sns.pairplot(sample_df, diag_kind="hist")
        plt.tight_layout()
        plt.savefig(out_dir / "pairplot_numeric.png")
        plt.close()

    # Mean rent per square meter by region
    if {"regio1", "rent_per_sqm"}.issubset(df.columns):
        by_region = (
            df.groupby("regio1")["rent_per_sqm"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(x=by_region.values, y=by_region.index, orient="h")
        plt.xlabel("Mean rent_per_sqm")
        plt.ylabel("regio1")
        plt.title("Top 15 regions by mean rent_per_sqm")
        plt.tight_layout()
        plt.savefig(out_dir / "rent_per_sqm_by_regio1.png")
        plt.close()


if __name__ == "__main__":
    output = build_dataset()
    print(f"Saved processed dataset to: {output}")

    # Run basic EDA on the cleaned dataset and save plots
    df_clean = pd.read_parquet(output)
    run_basic_eda(df_clean)
    print(f"Saved EDA plots to: {REPORTS_DIR}")

