import numpy as np

def create_windows(data, window_size=100):
    """
    Create overlapping sliding windows from time series data.

    Args:
        data (np.ndarray): Scaled time series data of shape (n_samples, n_features).
        window_size (int): Length of each window.

    Returns:
        np.ndarray: Array of shape (n_windows, window_size, n_features).
    """
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i : i + window_size])
    return np.array(windows)
