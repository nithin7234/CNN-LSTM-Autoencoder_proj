# utils/metrics.py
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def classification_report_from_labels(y_true, y_pred):
    """
    Compute precision, recall, f1, accuracy.
    y_true,y_pred: binary arrays
    """
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }
