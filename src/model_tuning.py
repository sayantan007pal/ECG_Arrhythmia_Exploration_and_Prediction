import os
import numpy as np
import pandas as pd
import wfdb
from datetime import datetime
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import bandpass_filter, normalize_signal
from feature_extraction import extract_features
from collections import Counter

# --- Data loading (same as main.py) ---
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

# --- Filter out rare classes (fewer than 3 samples) ---
counts = Counter(y)
min_samples = 3
mask = np.array([counts[label] >= min_samples for label in y])
X = X[mask]
y = y[mask]

# --- Optional: Subsample for tuning if dataset is large ---
TUNE_MAX = 2000  # max samples for tuning
if len(X) > TUNE_MAX:
    idx = np.random.choice(len(X), TUNE_MAX, replace=False)
    X_tune = X[idx]
    y_tune = y[idx]
else:
    X_tune = X
    y_tune = y

# --- Hyperparameter grids ---
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'SVM': {
        'C': [1],
        'kernel': ['rbf'],
        'gamma': ['scale']
    },
    'MLP': {
        'hidden_layer_sizes': [(50,)],
        'max_iter': [300],
        'activation': ['relu']
    }
}
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

# --- Tuning ---
best_score = 0
best_model = None
best_name = ''
best_params = None
results = []
for name, model in models.items():
    print(f'Tuning {name}...')
    n_jobs = -1 if name != 'SVM' else 1
    grid = GridSearchCV(model, param_grids[name], cv=2, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(X_tune, y_tune)
    acc = grid.best_score_
    print(f'Best {name} accuracy: {acc:.3f}')
    results.append({'model': name, 'accuracy': acc, 'params': grid.best_params_})
    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_
        best_name = name
        best_params = grid.best_params_

# --- Retrain best model on all data ---
best_model.fit(X, y)

# --- Save best model and results ---
version = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results/tuned_{version}'
os.makedirs(results_dir, exist_ok=True)
model_path = os.path.join(results_dir, f'best_model_{best_name}.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

# Save results summary
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, 'tuning_results.csv'), index=False)

# Save best params
with open(os.path.join(results_dir, 'best_params.txt'), 'w') as f:
    f.write(f'Best model: {best_name}\n')
    f.write(f'Best accuracy: {best_score:.3f}\n')
    f.write(f'Best params: {best_params}\n')

print(f'Best model: {best_name} (accuracy: {best_score:.3f})')
print(f'Model and results saved in: {results_dir}') 