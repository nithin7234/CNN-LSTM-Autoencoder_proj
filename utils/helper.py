# utils/helper.py
import os

def save_pickle(obj, path):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path):
    import joblib
    return joblib.load(path)
