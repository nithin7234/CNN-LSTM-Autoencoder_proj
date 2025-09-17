# Unsupervised Anomaly Detection - CNN-LSTM-Autoencoder

## Setup
1. Create virtual env:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows

2. Install requirements:
   pip install -r requirements.txt

3. Place dataset:
   proj/data/raw/equipment_anomaly_data.csv

4. Run:
   python main.py

## Project structure
See `config.py` for adjustable hyperparameters: WINDOW_SIZE, STRIDE, EPOCHS, etc.

## Notes
- This code assumes sensor numeric columns only.
- Thresholding uses mean+3*std on train reconstruction errors by default.
- If you have ground-truth labels (timestamps of anomalies), you can evaluate classification metrics by mapping window labels to timestamp labels (not included by default).
"# CNN-LSTM-Autoencoder_proj" 
