"""
Load a trained .joblib pipeline, run inference on JSON input, return predicted rent(s) in euros.

Usage:
  python -m scripts.inference --model models/rent_price_model.joblib --input request.json
  echo '{"livingSpace": 65, "noRooms": 3, ...}' | python -m scripts.inference --model models/rent_price_model.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from tools.train_model import SELECTED_FEATURE_COLS


def load_model(path: str | Path):
    import joblib
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def run_inference(pipeline, data: dict | list[dict], feature_cols: list[str] | None = None) -> list[float]:
    """
    Run inference on one or more records. Data can be a single dict or list of dicts.
    Returns list of predicted total rent in euros (model predicts log(rent), we apply exp).
    """
    if feature_cols is None:
        feature_cols = SELECTED_FEATURE_COLS
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)
    # Align columns: use only feature_cols that exist; fill missing with NaN for imputer
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    X = df[feature_cols]
    log_rent = pipeline.predict(X)
    rent_eur = np.exp(np.asarray(log_rent))
    return rent_eur.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run rent prediction from JSON input")
    parser.add_argument("--model", "-m", required=True, help="Path to .joblib model file")
    parser.add_argument("--input", "-i", default=None, help="Path to JSON file (one object or array); default: stdin")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file; default: stdout")
    args = parser.parse_args()

    pipeline = load_model(args.model)
    if args.input:
        with open(args.input) as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    predictions_eur = run_inference(pipeline, data)
    out = {"predictions_eur": predictions_eur}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
    else:
        json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
