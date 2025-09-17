import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_scaler(data: pd.DataFrame) -> StandardScaler:
    """Fit a new StandardScaler on the dataset."""
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

def save_scaler(scaler: StandardScaler, path: str):
    """Save the fitted scaler to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)

def load_scaler(path: str) -> StandardScaler:
    """Load an existing scaler from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)

def apply_scaler(data: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply scaler transform to the dataset."""
    return scaler.transform(data)

def load_or_fit_scaler(data: pd.DataFrame, path: str) -> StandardScaler:
    """Load scaler if exists, otherwise fit and save a new one."""
    if os.path.exists(path):
        print(f"Loading existing scaler from {path}...")
        return load_scaler(path)
    else:
        print("Fitting new scaler on data...")
        scaler = fit_scaler(data)
        save_scaler(scaler, path)
        return scaler
