# Data-Driven-Multi-Disease-Prediction-using-Machine-Learning

A machine learning system that predicts multiple diseases using medical parameters and provides real-time results.

## Project Structure
```plaintext
disease-prediction-system/
├── app.py                  # Main application interface
├── models/
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── parkinson_model.pkl
├── src/
│   ├── models.py          # Disease prediction models
│   └── train_models.py    # Model training scripts
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
└── requirements.txt
 ```
```

## Features
### Diabetes Prediction
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age
### Heart Disease Prediction
- Age & Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- ECG Results
- Heart Rate
- Exercise Data
- Vessel Information
### Parkinson's Disease Prediction
- Voice Parameters
- Frequency Measurements
- Amplitude Variables
- Speech Pattern Analysis
## Usage
1. Select disease type
2. Enter required medical parameters
3. Get instant prediction results
4. View prediction confidence
## Model Information
- Uses Random Forest Classification
- Trained on verified medical datasets
- High accuracy prediction
- Real-time processing
