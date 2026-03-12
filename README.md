# Rent Prediction

> [!NOTE]
> The deliverables are present in `notebook/rent_prediction_presentation.ipynb`.

Predict apartment **total rent** from German listing data using tabular features and optional LLM-derived signals. The pipeline cleans and featurizes ImmoScout-style data, optionally enriches descriptions with Gemini, and trains regression models (CatBoost, XGBoost, LightGBM, Random Forest).

---

## Overview

| Aspect | Details |
|--------|---------|
| **Data** | Apartment rental offers in Germany (region, size, rooms, condition, description, etc.) |
| **Target** | `totalRent` (stored as **log(totalRent)** in the processed dataset) |
| **Pipeline** | Raw data → clean & feature engineering → optional LLM enrichment → train/evaluate models |

---

## Repository structure

```
rent-prediction/
├── data/
│   ├── apartment-rental-offers-in-germany/   # Raw data: immo_data.csv (and optionally .parquet)
│   └── processed/                             # Cleaned & merged Parquet, EDA outputs
├── genai/                                     # Gemini API integration for text extraction
│   ├── config.py                              # API key, model name, system prompt
│   └── extractor.py                           # Batch extraction from descriptions
├── models/                                    # Saved .joblib model pipelines
├── notebooks/                                 # Jupyter notebooks (e.g. rent_prediction_presentation.ipynb)
├── reports/                                   # EDA plots (timestamped), training results
├── scripts/
│   ├── process_text_description.py           # LLM enrichment + merge → Parquet
│   ├── train_script.py                        # Multi-model, multi-hyperparameter training
│   └── train_feature_sets.py                  # Compare different feature sets
├── tools/
│   ├── data_pipeline.py                       # Load, clean, feature engineering, EDA
│   ├── train_model.py                         # Single run: train & save model
│   └── convert_to_parquet.py                  # CSV → Parquet (optional)
├── requirements.txt                           # Project dependencies
└── .env                                       # GEMINI_API_KEY (optional, for LLM step)
```

---

## Setup

1. **Clone and create a virtual environment**

   ```bash
   cd rent-prediction
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Data**

   Place the raw dataset under `data/apartment-rental-offers-in-germany/`:
   - `immo_data.csv` (required for the pipeline if you skip LLM step)

   Optional: convert CSV to Parquet in the same folder for faster I/O elsewhere:

   ```bash
   python -m tools.convert_to_parquet
   ```

4. **Optional: Gemini API (for LLM enrichment)**

   Create a `.env` in the project root:

   ```
   GEMINI_API_KEY=your_key_here
   ```

---

## Workflow

### 1. Data pipeline (clean + feature engineering)

Loads raw data (from `data/processed/immo_merged_with_llm_features.parquet` if present, else `data/apartment-rental-offers-in-germany/immo_data.csv`), cleans it, adds features, and writes the processed dataset plus EDA plots.

```bash
python -m tools.data_pipeline
```

- **Outputs**
  - Processed Parquet: path defined by `PROCESSED_DATASET` in `tools/data_pipeline.py` (default: `data/processed/immo_merged_with_llm_features_cleaned.parquet`)
  - EDA: `reports/eda/run_YYYYMMDD_HHMMSS/*.png` (correlation heatmap, pairplot, rent by region)

**Features added:** `rent_per_sqm`, `avg_rent_per_sqm_plz`, `avg_total_rent_regio3`, `avg_total_rent_regio2` (leave-one-out style to avoid leakage). Target is **log(totalRent)**.

### 2. Optional: LLM enrichment and merged dataset

Run Gemini on description/facilities/heating text to extract extra features, merge back, and save. Only rows with LLM results are kept in the merged file.

```bash
python -m scripts.process_text_description
```

- **Inputs:** `data/apartment-rental-offers-in-germany/immo_data.csv`
- **Outputs:** `data/genai_checkpoint.csv`, `data/processed/immo_merged_with_llm_features.parquet`

Then run the data pipeline (step 1); it will use the merged Parquet when present and write the cleaned version to `PROCESSED_DATASET`.

### 3. Train a model

**Single run (default: CatBoost, `SELECTED_FEATURE_COLS`):**

```bash
python -c "
from tools.train_model import train_rent_model, SELECTED_FEATURE_COLS
train_rent_model(feature_cols=SELECTED_FEATURE_COLS, run_name='my_run')
"
```

**Multi-model, multi-hyperparameter grid:**

```bash
python scripts/train_script.py
```

**Compare feature sets:**

```bash
python scripts/train_feature_sets.py
```

- **Outputs:** `models/rent_price_model_<run_name>.joblib` (or `models/rent_price_model.joblib`), and `reports/training_results.csv` / `reports/feature_set_results.csv` when using the scripts.

Training uses a **random sample** of the processed data by default (`max_rows=10_000`). Override with `max_rows` in `train_rent_model()` to use more or all rows.

---

## Configuration

- **Feature set:** `SELECTED_FEATURE_COLS` in `tools/train_model.py`
- **Default model:** `train_rent_model(..., model_name="catboost")`; alternatives: `"random_forest"`, `"xgboost"`, `"lightgbm"`
- **Hyperparameters:** Pass `model_params={...}` into `train_rent_model()` or edit `RUNS` / `FEATURE_SETS` in the scripts
- **LLM:** Prompt and model in `genai/config.py` (e.g. `SYSTEM_PROMPT`, `MODEL_NAME`)

---

## Notes

- Predictions from the saved pipeline are in **log(rent)**; convert back with `np.exp(pred)` for euros.
- LLM-derived features can be noisy; use with care or combine with rule-based checks.
- EDA plots are written under timestamped folders so previous runs are kept.
