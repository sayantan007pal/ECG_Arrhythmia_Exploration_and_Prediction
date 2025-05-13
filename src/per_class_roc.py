import os
import numpy as np
import pandas as pd
import wfdb
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
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
    except Exception as e:
        print(f"Error processing record {record_name}: {e}")
X = np.array(X)
y = np.array(y)

# Load best model
tuned_model_path, results_dir = find_latest_tuned_model()
with open(tuned_model_path, 'rb') as f:
    model = pickle.load(f)

# Use model.classes_ for correct class order
classes = model.classes_
y_bin = label_binarize(y, classes=classes)
n_classes = len(classes)

# Get prediction scores
if hasattr(model, 'predict_proba'):
    y_score = model.predict_proba(X)
else:
    y_score = model.decision_function(X)
    if y_score.ndim == 1:
        y_score = y_score.reshape(-1, 1)

# Plot ROC for each class
plt.figure(figsize=(10, 8))
roc_data = []
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC = {roc_auc:.2f})')
    roc_data.append(pd.DataFrame({'class': classes[i], 'fpr': fpr, 'tpr': tpr}))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Per-Class ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'per_class_roc.png'))
plt.show()
# Save ROC data
pd.concat(roc_data).to_csv(os.path.join(results_dir, 'per_class_roc_data.csv'), index=False)
print(f'Per-class ROC plot and data saved in: {results_dir}') 