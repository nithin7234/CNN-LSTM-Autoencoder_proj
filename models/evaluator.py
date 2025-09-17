import os
import numpy as np
from utils.plotter import plot_anomaly_scores

def evaluate_model(model, X_test, threshold=None, save_path="results/outputs/anomaly_scores.npy"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Reconstruct
    X_pred = model.predict(X_test)

    # Reconstruction errors
    errors = np.mean(np.mean((X_test - X_pred) ** 2, axis=-1), axis=-1)

    # Save scores
    np.save(save_path, errors)
    print(f"Anomaly scores saved to {save_path}")

    # Determine threshold if not given
    if threshold is None:
        threshold = np.mean(errors) + 3 * np.std(errors)

    # Classify anomalies
    anomalies = errors > threshold
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(errors)} samples.")

    # Plot results
    plot_anomaly_scores(errors, threshold, save_path.replace(".npy", ".png"))

    return errors, anomalies
