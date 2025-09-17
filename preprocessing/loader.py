# preprocessing/loader.py
import pandas as pd
import os

def load_dataset(path):
    """
    Load CSV into a pandas DataFrame.
    Expects time column or index — attempts to auto-detect.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    # Try to detect a datetime column
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
            except Exception:
                pass
    # If index is not datetime, keep as is. Ensure numeric columns only (drop any ID/text)
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric sensor columns found in dataset.")
    return numeric_df
