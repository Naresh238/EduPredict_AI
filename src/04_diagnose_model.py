from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("EDUPREDICT AI - MODEL DIAGNOSTICS")
print("=" * 65)

# Load the training and testing data
X_train = pd.read_csv(DATA_DIR / "X_train_raw.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_raw.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# 1. Establish a simple baseline
baseline = DummyRegressor(strategy="mean")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_predictions)
)
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\n1. BASELINE PERFORMANCE")
print("The baseline predicts the average training ML mark.")
print(f"Baseline MAE:  {baseline_mae:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline R2:   {baseline_r2:.4f}")

# 2. Inspect feature relationships in the original dataset
raw_file = ROOT / "data" / "raw" / "student_performance.csv"
df = pd.read_csv(raw_file)

numeric_df = df.select_dtypes(include=["number"])

correlations = numeric_df.corr()["ML Marks"].drop(
    labels=["ML Marks"],
    errors="ignore",
).sort_values(ascending=False)

correlation_file = REPORT_DIR / "feature_correlations_with_ml_marks.csv"
correlations.rename("Correlation_with_ML_Marks").to_csv(
    correlation_file,
    header=True,
)

print("\n2. NUMERIC FEATURE CORRELATIONS WITH ML MARKS")
print(correlations.round(4).to_string())

# 3. Save a concise diagnostic report
report_lines = [
    "EDUPREDICT AI - MODEL DIAGNOSTICS",
    "=" * 45,
    f"Baseline MAE: {baseline_mae:.4f}",
    f"Baseline RMSE: {baseline_rmse:.4f}",
    f"Baseline R2: {baseline_r2:.4f}",
    "",
    "Numeric feature correlations with ML Marks:",
    correlations.round(4).to_string(),
    "",
    "Interpretation:",
    "Correlations describe linear associations, not causation.",
    "A weak correlation does not rule out every nonlinear relationship.",
    "The dataset should be checked for realism and data quality.",
]

report_file = REPORT_DIR / "model_diagnostics.txt"
report_file.write_text("\n".join(report_lines), encoding="utf-8")

print("\n3. REPORTS SAVED")
print(correlation_file)
print(report_file)

print("\nMODEL DIAGNOSTICS COMPLETED SUCCESSFULLY!")
