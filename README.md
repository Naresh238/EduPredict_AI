# EduPredict AI – Student Performance Prediction System

An end-to-end Machine Learning project that predicts student academic performance and identifies students who may be at academic risk.

## Features

- Student grade prediction
- Academic risk classification
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Machine Learning model comparison
- Flask web application
- Prediction visualization
- Automated model testing

## Technologies

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- MySQL
- Plotly
- HTML and CSS

## Model Evaluation

Grade prediction:
- MAE: 2.9122
- RMSE: 3.9661
- R-squared: 0.2569

Risk classification:
- Accuracy: 69.62%
- Precision: 52.94%
- Recall: 69.23%
- F1-score: 60.00%
- ROC-AUC: 0.7271

These are experimental results on the selected test dataset.

## Installation

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

## Run Application

    python app.py

Open http://127.0.0.1:5000

## Run Tests

    python -m unittest discover -s tests -p "test_*.py" -v

## Dataset

The published Machine Learning models use the UCI Student Performance dataset.

Private datasets and individual prediction records are excluded from the repository.

## Disclaimer

This is an academic decision-support project. Predictions should not be used as the sole basis for decisions about students.

## Author

Developed by P. Naresh  
Saveetha University, Chennai
