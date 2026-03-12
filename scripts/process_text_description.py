import pandas as pd
import time
from pathlib import Path

from genai.extractor import GeminiFeatureExtractor

# Paths for merge step
RAW_CSV = Path("data/apartment-rental-offers-in-germany/immo_data.csv")
LLM_FEATURES_CSV = Path("data/genai_checkpoint.csv")
MERGED_PARQUET = Path("data/processed/immo_merged_with_llm_features.parquet")

def run_enrichment():
    # Load your data
    df = pd.read_csv("data/apartment-rental-offers-in-germany/immo_data.csv")
    extractor = GeminiFeatureExtractor()

    # Use scoutId from CSV to tag rows; fallback to row index if missing
    id_col = "scoutId" if "scoutId" in df.columns else None
    batch_size = 20  # smaller batches to avoid truncated JSON from the model
    results = []

    # Columns to send as 3 titled paragraphs per listing
    text_cols = ["description", "facilities", "heatingCosts"]
    titles = ["Description", "Facilities", "Heating costs"]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

    for i in range(154, 10000, batch_size): #len(df), batch_size):
        slice_df = df.iloc[i : i + batch_size]
        batch_ids = slice_df[id_col].tolist() if id_col else list(range(i, i + len(slice_df)))
        # Build 3 paragraphs with 3 titles per row
        batch_texts = []
        for _, row in slice_df.iterrows():
            parts = []
            for col, title in zip(text_cols, titles):
                val = row.get(col, "")
                if pd.isna(val) or val == "":
                    val = "Not specified"
                parts.append(f"{title}:\n{val}")
            batch_texts.append("\n\n".join(parts))
        print(f"Processing batch {i // batch_size}...")

        batch_output = extractor.process_batch(batch_texts)
        # Tag each result with serial/row index and original id for merging
        for j, out in enumerate(batch_output):
            row_id = batch_ids[j] if j < len(batch_ids) else i + j
            tagged = {"row_index": i + j, **(id_col and {id_col: row_id} or {})}
            if out is not None and isinstance(out, dict):
                tagged.update(out)
            else:
                tagged["_parse_failed"] = True
            results.append(tagged)

        temp_df = pd.DataFrame(results)
        temp_df.to_csv("data/genai_checkpoint.csv", index=False)

        time.sleep(4)  # Respect Gemini Free Tier RPM

    # Save LLM features
    final_features = pd.DataFrame(results)
    Path(LLM_FEATURES_CSV).parent.mkdir(parents=True, exist_ok=True)
    final_features.to_csv(LLM_FEATURES_CSV, index=False)
    print(f"Saved {LLM_FEATURES_CSV}")

    # Build merged Parquet: only rows present in llm_features, with LLM columns joined
    build_merged_csv()


def build_merged_csv(
    main_path: Path | str = RAW_CSV,
    llm_path: Path | str = LLM_FEATURES_CSV,
    out_path: Path | str = MERGED_PARQUET,
    on: str = "scoutId",
) -> Path:
    """
    Keep only rows that appear in llm_features; merge their LLM columns onto the main table; save as Parquet.
    """
    main_path = Path(main_path)
    llm_path = Path(llm_path)
    out_path = Path(out_path)
    if not main_path.exists():
        raise FileNotFoundError(f"Main data not found: {main_path}")
    if not llm_path.exists():
        raise FileNotFoundError(f"LLM features not found: {llm_path}")

    main_df = pd.read_csv(main_path)
    llm_df = pd.read_csv(llm_path)
    if on not in main_df.columns or on not in llm_df.columns:
        on = "row_index"
        if on not in llm_df.columns:
            raise ValueError("Need either 'scoutId' or 'row_index' in both main and LLM data to merge.")
        main_df = main_df.reset_index(names="row_index")

    # Inner merge: drop rows not in llm_features, add LLM columns
    merged = main_df.merge(llm_df, on=on, how="inner", suffixes=("", "_llm"))
    # Drop duplicate-named columns from LLM side if any
    merged = merged.loc[:, ~merged.columns.duplicated()]
    # Remove rows where LLM parse failed
    if "_parse_failed" in merged.columns:
        merged = merged[merged["_parse_failed"] != True]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"Merged Parquet saved: {out_path} ({len(merged)} rows)")
    return out_path


if __name__ == "__main__":
    run_enrichment()
    