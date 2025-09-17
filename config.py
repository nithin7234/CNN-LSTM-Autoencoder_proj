# config.py
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_CSV = os.path.join(RAW_DIR, "equipment_anomaly_data.csv")

# Model & results
RESULTS_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
REPORT_DIR = os.path.join(RESULTS_DIR, "reports")

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# Preprocessing & windowing
WINDOW_SIZE = 100        # number of timesteps per window (adjust to data)
STRIDE = 1               # sliding step
TEST_SIZE = 0.2          # fraction for test set
RANDOM_STATE = 42

# Scaling
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.save")

# Training
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "cnn_lstm_autoencoder.h5")

# Thresholding
THRESHOLD_STD_MULTIPLIER = 3.0  # threshold = mean + k * std of reconstruction error on train

# Misc
VERBOSE = 1
