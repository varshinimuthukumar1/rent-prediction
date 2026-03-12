"""Read two csv files and merge them by rows then save the merged file as a parquet file. The first csv file contains first 40k datapoints and the second csv file contains the rest of the datapoints."""
import pandas as pd
from pathlib import Path

csv1 = pd.read_csv("data/immo_data_new1.csv")
csv2 = pd.read_csv("data/immo_data_new2.csv")
merged = pd.concat([csv1, csv2])
merged.to_parquet("data/processed/immo_data_llm_externally_enriched.parquet")
