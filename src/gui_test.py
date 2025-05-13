import PySimpleGUI as sg
import os
import numpy as np
import pickle
import pandas as pd
from feature_extraction import extract_features
from datetime import datetime

# MIT-BIH arrhythmia code legend
ARRHYTHMIA_LEGEND = {
    'N': 'Normal beat',
    'L': 'Left bundle branch block beat',
    'R': 'Right bundle branch block beat',
    'e': 'Atrial escape beat',
    'j': 'Nodal (junctional) escape beat',
    'A': 'Atrial premature beat',
    'a': 'Aberrated atrial premature beat',
    'J': 'Nodal (junctional) premature beat',
    'S': 'Supraventricular premature beat',
    'V': 'Premature ventricular contraction',
    'E': 'Ventricular escape beat',
    'F': 'Fusion of ventricular and normal beat',
    '/': 'Paced beat',
    'f': 'Fusion of paced and normal beat',
    'Q': 'Unclassifiable beat',
    '[': 'Start of ventricular flutter/fibrillation',
    ']': 'End of ventricular flutter/fibrillation',
    '!': 'Ventricular flutter wave',
    'x': 'Non-conducted P-wave (blocked APC)',
    '|': 'Isolated QRS-like artifact',
}

LEGEND_TEXT = '\n'.join([f"{k}: {v}" for k, v in ARRHYTHMIA_LEGEND.items()])

SAMPLE_ECG = ', '.join([str(np.round(np.sin(2 * np.pi * i / 60), 2)) for i in range(30)])

# Helper to list available models
MODEL_DIR = 'results'
def list_models():
    model_files = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.startswith('model_') and file.endswith('.pkl'):
                model_files.append(os.path.join(root, file))
            if file.startswith('best_model_') and file.endswith('.pkl'):
                model_files.append(os.path.join(root, file))
    return model_files

# Info/help popup
INFO_TEXT = (
    "Input Format:\n"
    "- Enter a single ECG segment as comma-separated values (e.g., 0.1, 0.2, -0.1, ...).\n"
    "- Or upload a CSV file with one segment per row.\n"
    "- Typical ECG values range from -2 to 2 mV.\n"
    "- Segment length (number of samples) is customizable.\n\n"
    "Prediction Codes (MIT-BIH):\n" + LEGEND_TEXT +
    "\n\nThe model analyzes the shape and timing of your ECG signal to predict the type of heartbeat."
)

# GUI layout
def main():
    sg.theme('LightBlue')
    layout = [
        [sg.Text('ECG Arrhythmia Model Tester', font=('Any', 16))],
        [sg.Text('Select Model:'), sg.Combo(list_models(), key='MODEL', size=(60, 1)), sg.Button('Refresh Models', key='REFRESH_MODELS')],
        [sg.Text('Sample ECG input (first 30 values):'), sg.Text(SAMPLE_ECG, font=('Any', 8), size=(80,1))],
        [sg.Text('Segment Length (samples):'), sg.Input('360', key='SEG_LEN', size=(6,1)), sg.Text('Typical: 360 for 1s at 360Hz')],
        [sg.Text('Enter ECG segment (comma-separated values):')],
        [sg.Multiline('', size=(60, 3), key='MANUAL_ECG', tooltip='Comma-separated values, e.g., 0.1, 0.2, -0.1, ...')],
        [sg.Text('Or upload CSV file (one segment per row):'), sg.Input(key='CSV_PATH'), sg.FileBrowse()],
        [sg.Text('Sampling Frequency (Hz):'), sg.Input('360', key='FS', size=(6,1))],
        [sg.Text('Optional: Enter actual label for accuracy comparison:'), sg.Input('', key='ACTUAL_LABEL', size=(10,1))],
        [sg.Button('Predict'), sg.Button('Clear'), sg.Button('Help'), sg.Exit()],
        [sg.Text('Prediction:'), sg.Text('', key='PREDICTION', size=(40,1))],
        [sg.Text('Accuracy:'), sg.Text('', key='ACCURACY', size=(20,1))],
        [sg.Text('Saved results file:'), sg.Text('', key='RESULTS_FILE', size=(40,1))],
        [sg.Text('Prediction Code Legend:')],
        [sg.Multiline(LEGEND_TEXT, size=(60, 10), disabled=True, font=('Any', 9))],
        [sg.Text('What does the model do? The model analyzes the shape and timing of your ECG signal to predict the type of heartbeat. See legend above for code meanings.')]
    ]
    window = sg.Window('ECG Arrhythmia Model Tester', layout, finalize=True)
    results = []
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'REFRESH_MODELS':
            window['MODEL'].update(values=list_models())
            continue
        if event == 'Help':
            sg.popup_scrolled(INFO_TEXT, title='Help / Info', size=(80, 20))
            continue
        if event == 'Clear':
            window['MANUAL_ECG'].update('')
            window['CSV_PATH'].update('')
            window['PREDICTION'].update('')
            window['ACCURACY'].update('')
            window['RESULTS_FILE'].update('')
            window['ACTUAL_LABEL'].update('')
            continue
        if event == 'Predict':
            model_path = values['MODEL']
            fs = int(values['FS'])
            seg_len = int(values['SEG_LEN'])
            actual_label = values['ACTUAL_LABEL']
            if not model_path or not os.path.exists(model_path):
                sg.popup_error('Please select a valid model file.')
                continue
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            # Get ECG data
            segments = []
            if values['MANUAL_ECG'].strip():
                try:
                    arr = np.array([float(x) for x in values['MANUAL_ECG'].replace('\n',',').split(',') if x.strip()])
                    if len(arr) != seg_len:
                        sg.popup_error(f'Input segment length ({len(arr)}) does not match expected ({seg_len}).')
                        continue
                    segments.append(arr)
                except Exception as e:
                    sg.popup_error(f'Error parsing manual ECG input: {e}')
                    continue
            if values['CSV_PATH']:
                try:
                    df = pd.read_csv(values['CSV_PATH'], header=None)
                    for row in df.values:
                        if len(row) != seg_len:
                            sg.popup_error(f'CSV row length ({len(row)}) does not match expected ({seg_len}).')
                            continue
                        segments.append(np.array(row, dtype=float))
                except Exception as e:
                    sg.popup_error(f'Error reading CSV: {e}')
                    continue
            if not segments:
                sg.popup_error('Please enter ECG data manually or upload a CSV file.')
                continue
            # Predict
            preds = []
            for seg in segments:
                feats = extract_features(seg, fs)
                feats = feats.reshape(1, -1)
                pred = model.predict(feats)[0]
                preds.append(pred)
            window['PREDICTION'].update(', '.join(map(str, preds)))
            # Accuracy if actual label provided
            acc = ''
            if actual_label:
                acc = f"{np.mean([p==actual_label for p in preds])*100:.1f}%"
                window['ACCURACY'].update(acc)
            # Save results with legend
            results_file = os.path.join(os.path.dirname(model_path), f'gui_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv')
            df_out = pd.DataFrame({'input': [seg.tolist() for seg in segments], 'prediction': preds, 'actual': [actual_label]*len(preds)})
            df_out.to_csv(results_file, index=False)
            # Save legend as a separate CSV for clarity
            legend_file = os.path.join(os.path.dirname(model_path), 'arrhythmia_legend.csv')
            pd.DataFrame(list(ARRHYTHMIA_LEGEND.items()), columns=['Code', 'Description']).to_csv(legend_file, index=False)
            window['RESULTS_FILE'].update(results_file)
    window.close()

if __name__ == '__main__':
    main() 