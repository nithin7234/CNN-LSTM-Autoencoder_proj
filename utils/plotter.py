import matplotlib.pyplot as plt

def plot_loss(history, save_path="results/plots/training_loss.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training loss plot saved to {save_path}")

def plot_anomaly_scores(errors, threshold, save_path="results/plots/anomaly_scores.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(threshold, color="r", linestyle="--", label=f"Threshold ({threshold:.4f})")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Anomaly scores plot saved to {save_path}")
