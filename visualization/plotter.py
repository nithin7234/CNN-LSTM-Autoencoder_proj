import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def plot_anomaly_scores(scores, threshold, output_path, show=True):
    """
    Plot anomaly scores with threshold line.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(scores, label="Reconstruction Error", color="blue")
    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Anomaly Scores over Time")
    plt.xlabel("Sample Index")
    plt.ylabel("Score")
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_reconstruction_error_distribution(scores, output_path, show=True):
    """
    Histogram of reconstruction errors.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, bins=50, kde=True, color="purple")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_anomalies_on_series(series, scores, anomalies, output_path, show=True):
    """
    Plot time series with anomalies marked.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Series", color="blue")
    plt.scatter(np.where(anomalies)[0], series[anomalies], color="red", label="Anomalies")
    plt.legend()
    plt.title("Time Series with Detected Anomalies")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def plot_boxplot_errors(scores, output_path, show=True):
    """
    Boxplot of reconstruction errors to visualize spread/outliers.
    """
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=scores, color="lightblue")
    plt.title("Boxplot of Reconstruction Errors")
    plt.ylabel("Error")
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()

def save_results_table(scores, anomalies, output_path):
    """
    Save anomalies as a CSV summary.
    """
    df = pd.DataFrame({
        "Index": np.arange(len(scores)),
        "AnomalyScore": scores,
        "Anomaly": anomalies.astype(int)
    })
    df.to_csv(output_path, index=False)
