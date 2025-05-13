import os
import numpy as np
import pandas as pd
import wfdb
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from preprocessing import bandpass_filter, normalize_signal
from feature_extraction import extract_features

def find_latest_tuned_model():
    results_root = 'results'
    tuned_dirs = [d for d in os.listdir(results_root) if d.startswith('tuned_')]
    if not tuned_dirs:
        raise FileNotFoundError('No tuned_ results directory found.')
    latest_dir = sorted(tuned_dirs)[-1]
    model_files = [f for f in os.listdir(os.path.join(results_root, latest_dir)) if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError('No best_model_*.pkl found in latest tuned_ directory.')
    return os.path.join(results_root, latest_dir, model_files[0]), os.path.join(results_root, latest_dir)

# Load data
DATA_DIR = 'data/mit-bih-arrhythmia-database-1.0.0'
RECORDS_FILE = os.path.join(DATA_DIR, 'RECORDS')
fs = 360
X = []
y = []
segments = []
skipped_segments = 0
with open(RECORDS_FILE, 'r') as f:
    record_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
for record_name in record_names:
    try:
        record_path = os.path.join(DATA_DIR, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        ecg_signal = record.p_signal[:, 0]
        for i, sample in enumerate(annotation.sample):
            win_size = int(0.5 * fs)
            start = max(sample - win_size, 0)
            end = min(sample + win_size, len(ecg_signal))
            segment = ecg_signal[start:end]
            if len(segment) < 2 * win_size or np.all(segment == 0):
                skipped_segments += 1
                continue
            filtered = bandpass_filter(segment, 0.5, 40, fs)
            normed = normalize_signal(filtered)
            features = extract_features(normed, fs)
            if features.size == 0 or np.any(np.isnan(features)):
                skipped_segments += 1
                continue
            X.append(features)
            y.append(annotation.symbol[i])
            segments.append(segment)
    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
X = np.array(X)
y = np.array(y)

# Load best model
tuned_model_path, results_dir = find_latest_tuned_model()
with open(tuned_model_path, 'rb') as f:
    model = pickle.load(f)

# Predict
y_pred = model.predict(X)

# Find misclassified examples
mis_idx = np.where(y_pred != y)[0]
mis_df = pd.DataFrame({
    'true_label': y[mis_idx],
    'predicted_label': y_pred[mis_idx],
    'features': [X[i].tolist() for i in mis_idx]
})
mis_df.to_csv(os.path.join(results_dir, 'misclassified_examples.csv'), index=False)

# Plot a few misclassified signals
plt.figure(figsize=(12, 8))
for i, idx in enumerate(mis_idx[:5]):
    plt.subplot(5, 1, i+1)
    plt.plot(segments[idx])
    plt.title(f'True: {y[idx]}, Predicted: {y_pred[idx]}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'misclassified_signals.png'))
plt.show()
print(f'Misclassified examples and plots saved in: {results_dir}') 