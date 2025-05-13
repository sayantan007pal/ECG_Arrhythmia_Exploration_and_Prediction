# ECG Arrhythmia Detection using Machine Learning

This project implements an automated system for detecting cardiac arrhythmias using machine learning techniques on ECG signals. The system processes ECG data, extracts relevant features, and classifies different types of arrhythmias.

## Features
- ECG signal preprocessing and filtering
- Feature extraction from ECG signals
- Multiple machine learning models for arrhythmia classification
- Model evaluation and visualization tools
- Support for MIT-BIH Arrhythmia Database format

## Project Structure
```
ECG_Arrhythmia_Exploration_and_Prediction
├── data/
│   └── mit-bih-arrhythmia-database-1.0.0/
│       ├── mitdbdir/
│       │   ├── samples/
│       │   └── src/
│       └── x_mitdb/
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   ├── evaluate.py
│   ├── main.py
│   ├── gui_test.py
│   ├── model_tuning.py
│   ├── tsne_visualization.py
│   ├── per_class_roc.py
│   └── error_analysis.py
├── requirements.txt
└── README.md
```

## Setup
1. Clone the repository and navigate to the project directory.
   ```bash
   git clone https://github.com/sayantan007pal/ECG_Arrhythmia_Exploration_and_Prediction.git
   cd ECG_Arrhythmia_Exploration_and_Prediction

   ```
  - Create Virtual environmet
   ```bash
    python -m venv venv
    source venv/bin/activate
   ```
    
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the MIT-BIH Arrhythmia Database files in the `data/mit-bih-arrhythmia-database-1.0.0/` directory as shown above.

## Usage

### Main Pipeline
Run the main pipeline to preprocess data, extract features, train models, and evaluate results:
```bash
python src/main.py
```
- **What it does:** Loads ECG data, preprocesses and extracts features, trains multiple models, and saves results/plots in a timestamped `results/` folder.
- **Outputs:**
  - Model accuracy, F1 score, and confusion matrices (as images and CSVs)
  - Feature importance plots (for Random Forest)
  - Class distribution plots

### Interactive GUI for Model Testing
Test your trained models interactively:
```bash
python src/gui_test.py
```
<img width="1352" alt="Screenshot 2025-05-13 at 3 54 36 AM" src="https://github.com/user-attachments/assets/b1ad6735-cf13-48fd-87f2-2cb640748853" />

- **What it does:**
  - Lets you select a saved model (RandomForest, SVM, MLP)
  - Enter ECG segment data manually (comma-separated) or upload a CSV file (one segment per row)
  - Set segment length and sampling frequency
  - See the model's prediction(s) and optionally enter the actual label to compare and display accuracy
  - Saves prediction results and a legend explaining all arrhythmia codes
- **Input format:**
  - Comma-separated ECG values (e.g., `0.1, 0.2, -0.1, ...`)
  - Typical segment length: 360 (for 1 second at 360 Hz), but customizable
  - CSV: Each row is one segment
- **Prediction codes:**
  - The GUI and output CSV include a legend for all MIT-BIH arrhythmia codes (e.g., N: Normal beat, V: Premature ventricular contraction, etc.)

### Model Hyperparameter Tuning
Tune model hyperparameters for best accuracy:
```bash
python src/model_tuning.py
```
- **What it does:**
  - Performs hyperparameter tuning for RandomForest, SVM, and MLP using GridSearchCV (with subsampling for speed)
  - Selects the best model (highest cross-validated accuracy)
  - Retrains the best model on all data
  - Saves the best model, its parameters, and a summary of all tuning results in a timestamped `results/tuned_<timestamp>/` folder
- **Outputs:**
  - Best model as a `.pkl` file
  - Tuning results and best parameters as CSV and TXT

### Advanced Visualizations & Analysis
#### t-SNE Visualization
```bash
python src/tsne_visualization.py
```
![Figure_1](https://github.com/user-attachments/assets/3391cd4c-79d5-49ae-85ac-3ce2be702332)

- **What it does:**
  - Runs t-SNE on extracted features to reduce to 2D
  - Plots a 2D scatter plot colored by arrhythmia class
  - Saves the plot and the t-SNE embedding as CSV in a timestamped results folder
- **How to interpret:**
  - Points close together have similar ECG features
  - Clusters may correspond to different arrhythmia types

#### Per-Class ROC Curves
```bash
python src/per_class_roc.py
```
![Per-Class-ROC-Curves](https://github.com/user-attachments/assets/021915ad-4234-4640-b1d8-e2bddec50ee3)

- **What it does:**
  - Loads the best tuned model
  - Plots ROC curves for each class (multi-class ROC)
  - Saves the plot and ROC data as CSV in the latest tuning results folder
- **How to interpret:**
  - Each curve shows the model's ability to distinguish one arrhythmia type from others
  - AUC (Area Under Curve) closer to 1 means better discrimination

#### Error Analysis
```bash
python src/error_analysis.py
```
![True_A_Pridicted_N](https://github.com/user-attachments/assets/a07660ad-9292-488a-8fca-932c3ab8b5b5)

- **What it does:**
  - Loads the best tuned model
  - Finds and saves all misclassified examples (true label, predicted label, features)
  - Plots a few example misclassified ECG signals
  - Saves all outputs in the latest tuning results folder
- **How to interpret:**
  - Review misclassified signals to understand model weaknesses
  - Use plots to visually inspect where the model struggles

## Arrhythmia Code Legend (MIT-BIH)
| Code | Description |
|------|-------------------------------------------------------------|
| N    | Normal beat                                                |
| L    | Left bundle branch block beat                              |
| R    | Right bundle branch block beat                             |
| e    | Atrial escape beat                                         |
| j    | Nodal (junctional) escape beat                             |
| A    | Atrial premature beat                                      |
| a    | Aberrated atrial premature beat                            |
| J    | Nodal (junctional) premature beat                          |
| S    | Supraventricular premature beat                            |
| V    | Premature ventricular contraction                          |
| E    | Ventricular escape beat                                    |
| F    | Fusion of ventricular and normal beat                      |
| /    | Paced beat                                                 |
| f    | Fusion of paced and normal beat                            |
| Q    | Unclassifiable beat                                        |
| [    | Start of ventricular flutter/fibrillation                  |
| ]    | End of ventricular flutter/fibrillation                    |
| !    | Ventricular flutter wave                                   |
| x    | Non-conducted P-wave (blocked APC)                         |
| |    | Isolated QRS-like artifact                                 |

These codes are standard in the MIT-BIH Arrhythmia Database and are explained in the GUI and output files.

## Customization
- Modify `src/models.py` to add or change machine learning models.
- Update `src/feature_extraction.py` for custom ECG features.

## License
MIT License 
