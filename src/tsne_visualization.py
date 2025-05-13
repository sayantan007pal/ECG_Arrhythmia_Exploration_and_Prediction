import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from datetime import datetime

# Load features and labels (assume last run's results or rerun extraction here)
DATA_DIR = 'data/mit-bih-arrhythmia-database-1.0.0'
RECORDS_FILE = os.path.join(DATA_DIR, 'RECORDS')
fs = 360
X = []
y = []
skipped_segments = 0
import wfdb
from preprocessing import bandpass_filter, normalize_signal
from feature_extraction import extract_features

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

# t-SNE
print('Running t-SNE...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X)

# Save embedding
version = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = f'results/tsne_{version}'
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame({'x': X_embedded[:,0], 'y': X_embedded[:,1], 'label': y}).to_csv(f'{results_dir}/tsne_embedding.csv', index=False)

# Plot
df_plot = pd.DataFrame({'x': X_embedded[:,0], 'y': X_embedded[:,1], 'label': y})
plt.figure(figsize=(10,8))
sns.scatterplot(data=df_plot, x='x', y='y', hue='label', palette='tab10', s=20, alpha=0.7)
plt.title('t-SNE of ECG Features by Arrhythmia Class')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{results_dir}/tsne_plot.png')
plt.show()
print(f't-SNE plot and embedding saved in: {results_dir}') 