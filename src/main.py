import os
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from preprocessing import bandpass_filter, normalize_signal
from feature_extraction import extract_features
from models import train_and_evaluate
from evaluate import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# --- Data loading using MIT-BIH format and wfdb ---
DATA_DIR = 'data/mit-bih-arrhythmia-database-1.0.0'
RECORDS_FILE = os.path.join(DATA_DIR, 'RECORDS')
fs = 360  # Sampling frequency (Hz), adjust if needed

X = []  # Feature vectors
y = []  # Labels
skipped_segments = 0

# Versioning
version = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results/run_{version}'
os.makedirs(results_dir, exist_ok=True)

# Read all record names from the RECORDS file
with open(RECORDS_FILE, 'r') as f:
    record_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

for record_name in record_names:
    try:
        record_path = os.path.join(DATA_DIR, record_name)
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        ecg_signal = record.p_signal[:, 0]  # Use first channel
        # For each annotation, extract a window around the R-peak
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

if skipped_segments > 0:
    print(f"Skipped {skipped_segments} segments due to being too short, all zeros, or invalid features.")

X = np.array(X)
y = np.array(y)

# --- Visualization: Class Distribution ---
plt.figure(figsize=(10, 4))
class_counts = pd.Series(y).value_counts().sort_index()
class_counts.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Arrhythmia Symbol')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f'{results_dir}/class_distribution.png')
plt.close()

# --- Model training and evaluation ---
results = train_and_evaluate(X, y)

# --- Save metrics and confusion matrices ---
metrics_list = []
for name, res in results.items():
    metrics_list.append({
        'model': name,
        'accuracy': res['accuracy'],
        'f1_score': res['f1_score']
    })
    cm = res['confusion_matrix']
    n_classes = cm.shape[0]
    unique_labels = list(sorted(set(y)))
    if n_classes != len(unique_labels):
        class_names = [str(i) for i in range(n_classes)]
    else:
        class_names = unique_labels
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f'{results_dir}/confusion_matrix_{name}.csv')
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names=class_names, title=f'{name} Confusion Matrix')
    plt.savefig(f'{results_dir}/confusion_matrix_{name}.png')
    plt.close()
    # Save model as pickle
    with open(f'{results_dir}/model_{name}.pkl', 'wb') as f:
        pickle.dump(res['model'], f)

# Save metrics summary
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(f'{results_dir}/metrics_summary.csv', index=False)

# --- Feature Importance (Random Forest) ---
if 'RandomForest' in results:
    rf = results['RandomForest']['model']
    if hasattr(rf, 'feature_importances_'):
        feat_names = [
            'Mean RR', 'Std RR', 'Min RR', 'Max RR', 'Num RR',
            'Energy', 'Skewness', 'Kurtosis', 'Peak2Peak', 'MAD', 'MedAD'
        ]
        importances = rf.feature_importances_
        plt.figure(figsize=(10, 4))
        sns.barplot(x=importances, y=feat_names)
        plt.title('Random Forest Feature Importances')
        plt.tight_layout()
        plt.savefig(f'{results_dir}/rf_feature_importance.png')
        plt.close()

print(f"Results saved in: {results_dir}") 