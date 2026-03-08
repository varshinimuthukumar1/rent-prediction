from pathlib import Path

import pandas as pd


def csv_to_parquet(csv_path: str | Path, parquet_path: str | Path | None = None) -> Path:
    """
    Load a CSV file and save it as a Parquet file.

    If parquet_path is not provided, the Parquet file will be written
    next to the CSV with the same stem name.
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")
    else:
        parquet_path = Path(parquet_path)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, index=False)

    print(f"Saved Parquet file to: {parquet_path}")
    return parquet_path


if __name__ == "__main__":
    # Default: convert the Kaggle dataset CSV shipped with this project
    default_csv = Path("data/apartment-rental-offers-in-germany/immo_data.csv")
    if not default_csv.exists():
        raise SystemExit(f"CSV file not found: {default_csv.resolve()}")

    csv_to_parquet(default_csv)

