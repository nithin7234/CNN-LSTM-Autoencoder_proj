# main.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model as tf_load_model  # fallback

# Project imports (adjust if your module names are slightly different)
from preprocessing.loader import load_dataset
from preprocessing.scaler import load_or_fit_scaler, apply_scaler
from preprocessing.windowing import create_windows
from models.cnn_lstm_autoencoder import build_cnn_lstm_autoencoder
from models.trainer import train_model, save_model, load_model

# --- Config ---
DATA_PATH = "data/equipment_anomaly_data.csv"
SCALER_PATH = "results/scaler.pkl"
MODEL_PATH = "results/models/cnn_lstm_autoencoder.keras"
OUTPUT_DIR = "results/outputs"

WINDOW_SIZE = 100
EPOCHS = 5         # reduce for quick tests
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# small helper plots (self-contained)
import matplotlib.pyplot as plt
def save_scores_plot(scores, threshold, path):
    plt.figure(figsize=(10,4))
    plt.plot(scores, label="Reconstruction error")
    plt.axhline(threshold, color="r", linestyle="--", label=f"threshold={threshold:.4f}")
    plt.legend()
    plt.title("Anomaly Scores")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved plot: {path}")

def save_hist_plot(scores, path):
    plt.figure(figsize=(8,4))
    plt.hist(scores, bins=60)
    plt.title("Reconstruction Error Distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved histogram: {path}")

# ---------------- MAIN ----------------
def main():
    # 1) Load dataset
    print("[INFO] Loading dataset...")
    df = load_dataset(DATA_PATH)   # expects numeric-only DataFrame (preprocessing/loader.py)

    # 2) Load or fit scaler
    print("[INFO] Loading or fitting scaler...")
    scaler = load_or_fit_scaler(df, SCALER_PATH)  # function from preprocessing/scaler.py
    data_scaled = apply_scaler(df, scaler)        # returns numpy array shaped (n_samples, n_features)

    # 3) Create sliding windows -> X shape: (n_windows, window_size, n_features)
    print("[INFO] Creating sliding windows...")
    X = create_windows(data_scaled, window_size=WINDOW_SIZE)
    if X.size == 0:
        raise RuntimeError(f"Not enough data to create any windows. Need at least WINDOW_SIZE={WINDOW_SIZE} rows.")

    print(f"[INFO] Windows created: {X.shape}")

    # 4) Train / validation split
    X_train, X_val = train_test_split(X, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print(f"[INFO] X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 5) Build or load model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print("[WARN] models.trainer.load_model failed, trying tf.keras.models.load_model as fallback:", e)
            model = tf_load_model(MODEL_PATH, compile=False)
            print("[INFO] Loaded model via tf keras fallback.")
    else:
        timesteps, n_features = X.shape[1], X.shape[2]
        print(f"[INFO] Building new model with timesteps={timesteps}, n_features={n_features} ...")
        model = build_cnn_lstm_autoencoder(timesteps, n_features)

        # 6) Train model (trainer supports optional x_val)
        print("[INFO] Training model...")
        try:
            history = train_model(model, X_train, X_val, epochs=EPOCHS, batch_size=BATCH_SIZE, save_path=MODEL_PATH)
        except TypeError:
            # older trainer signature might not accept x_val; fallback to calling without x_val
            print("[WARN] trainer.train_model rejected x_val parameter — retrying without it.")
            history = train_model(model, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, save_path=MODEL_PATH)

        # ensure model saved in .keras
        if not os.path.exists(MODEL_PATH):
            try:
                save_model(model, MODEL_PATH)
            except Exception as e:
                print("[WARN] save_model failed:", e)

    # 7) Evaluate on validation set -> anomaly scores
    print("[INFO] Predicting on validation set...")
    X_val_pred = model.predict(X_val)
    # per-window MSE across timesteps and features
    scores = np.mean(np.mean((X_val - X_val_pred) ** 2, axis=2), axis=1)

    # save scores
    np.save(os.path.join(OUTPUT_DIR, "anomaly_scores.npy"), scores)
    # threshold
    threshold = float(np.mean(scores) + 3 * np.std(scores))
    anomalies = scores > threshold
    n_anomalies = int(np.sum(anomalies))
    print(f"[INFO] Detected {n_anomalies} anomalies out of {len(scores)} windows (threshold={threshold:.6f}).")

    # save CSV summary
    summary_df = pd.DataFrame({"score": scores, "is_anomaly": anomalies.astype(int)})
    summary_csv_path = os.path.join(OUTPUT_DIR, "anomaly_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[INFO] Saved summary CSV: {summary_csv_path}")

    # save plots
    save_scores_plot(scores, threshold, os.path.join(OUTPUT_DIR, "anomaly_scores.png"))
    save_hist_plot(scores, os.path.join(OUTPUT_DIR, "error_distribution.png"))

    print("[INFO] All done. Results in:", OUTPUT_DIR)

    from visualization.plotter import (
    plot_anomaly_scores,
    plot_reconstruction_error_distribution,
    plot_anomalies_on_series,
    plot_boxplot_errors,
    save_results_table
    )

    # Example usage after anomaly detection
    plot_anomaly_scores(scores, threshold, "results/outputs/anomaly_scores.png")
    plot_reconstruction_error_distribution(scores, "results/outputs/error_distribution.png")
    plot_anomalies_on_series(X_val[:, 0, 0], scores, anomalies, "results/outputs/anomalies_series.png")
    plot_boxplot_errors(scores, "results/outputs/error_boxplot.png")

    save_results_table(scores, anomalies, "results/outputs/anomalies_summary.csv")


if __name__ == "__main__":
    main()
