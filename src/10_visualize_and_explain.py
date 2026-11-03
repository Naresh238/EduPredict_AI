from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "uci_student_performance" / "extracted"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EDUPREDICT AI - VISUALIZATION AND EXPLAINABILITY")
print("=" * 70)

# --------------------------------------------------
# 1. STUDENT RISK DISTRIBUTION
# --------------------------------------------------
df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
df = df.drop_duplicates().copy()

risk_counts = np.where(df["G3"] < 10, "At Risk", "Not At Risk")
risk_series = pd.Series(risk_counts).value_counts()

plt.figure(figsize=(8, 5))
risk_series.plot(kind="bar")
plt.title("Student Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "student_risk_distribution.png", dpi=200)
plt.close()

# --------------------------------------------------
# 2. REGRESSION MODEL COMPARISON
# --------------------------------------------------
regression_file = REPORT_DIR / "uci_model_comparison.csv"

if regression_file.exists():
    results = pd.read_csv(regression_file)
    results = results.sort_values("Test_MAE")

    plt.figure(figsize=(10, 6))
    plt.bar(results["Model"], results["Test_MAE"])
    plt.title("Regression Model Comparison - Test MAE")
    plt.xlabel("Model")
    plt.ylabel("Mean Absolute Error (lower is better)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "regression_model_comparison.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 3. CLASSIFICATION MODEL COMPARISON
# --------------------------------------------------
classification_file = REPORT_DIR / "uci_risk_model_comparison.csv"

if classification_file.exists():
    results = pd.read_csv(classification_file)
    results = results[results["Model"] != "Dummy Baseline"]
    results = results.sort_values("CV_F1", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(results["Model"], results["CV_F1"])
    plt.title("Risk Classification - Cross-Validation F1")
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "classification_model_comparison.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 4. CONFUSION MATRIX
# --------------------------------------------------
X = df.drop(columns=["G1", "G2", "G3"])
y = (df["G3"] < 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

risk_model_path = MODEL_DIR / "uci_best_risk_model.joblib"
risk_model = joblib.load(risk_model_path)

plt.figure(figsize=(7, 6))
ConfusionMatrixDisplay.from_estimator(
    risk_model,
    X_test,
    y_test,
    display_labels=["Not At Risk", "At Risk"],
    cmap="Blues",
    values_format="d",
)
plt.title("Student Risk Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "risk_confusion_matrix.png", dpi=200)
plt.close()

# --------------------------------------------------
# 5. ROC CURVE
# --------------------------------------------------
plt.figure(figsize=(7, 6))
RocCurveDisplay.from_estimator(
    risk_model,
    X_test,
    y_test,
)
plt.title("Student Risk Classification - ROC Curve")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "risk_roc_curve.png", dpi=200)
plt.close()

# --------------------------------------------------
# 6. PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------
print("\nCalculating permutation feature importance...")
print("This may take a short time.")

importance_result = permutation_importance(
    risk_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": importance_result.importances_mean,
    "Importance_Std": importance_result.importances_std,
})

importance_df = importance_df.sort_values(
    "Importance_Mean",
    ascending=False,
)

importance_df.to_csv(
    REPORT_DIR / "uci_risk_permutation_importance.csv",
    index=False,
)

top_features = importance_df.head(15).sort_values("Importance_Mean")

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"],
    top_features["Importance_Mean"],
    xerr=top_features["Importance_Std"],
)
plt.title("Top 15 Features - Permutation Importance")
plt.xlabel("Decrease in F1-score when feature is shuffled")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "risk_feature_importance.png", dpi=200)
plt.close()

# --------------------------------------------------
# 7. ACTUAL VS PREDICTED GRADES
# --------------------------------------------------
prediction_file = REPORT_DIR / "uci_test_predictions.csv"

if prediction_file.exists():
    predictions_df = pd.read_csv(prediction_file)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        predictions_df["Actual_G3"],
        predictions_df["Predicted_G3"],
    )
    plt.plot([0, 20], [0, 20], linestyle="--")
    plt.title("Actual vs Predicted Mathematics Grades")
    plt.xlabel("Actual Grade")
    plt.ylabel("Predicted Grade")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted_grades.png", dpi=200)
    plt.close()

# --------------------------------------------------
# 8. SAVE A SUMMARY
# --------------------------------------------------
summary_lines = [
    "EDUPREDICT AI - VISUALIZATION REPORT",
    "=" * 50,
    "Charts generated:",
    "1. student_risk_distribution.png",
    "2. regression_model_comparison.png",
    "3. classification_model_comparison.png",
    "4. risk_confusion_matrix.png",
    "5. risk_roc_curve.png",
    "6. risk_feature_importance.png",
    "7. actual_vs_predicted_grades.png",
    "",
    "Feature importance is measured using permutation importance.",
    "A feature's importance reflects its effect on test-set F1-score",
    "when its values are shuffled. It does not prove causation.",
]

(REPORT_DIR / "uci_visualization_report.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print("\nVISUALIZATION GENERATION COMPLETED!")
print("\nGenerated charts:")

for file in sorted(FIGURE_DIR.glob("*.png")):
    print(file)

print("\nFeature importance report:")
print(REPORT_DIR / "uci_risk_permutation_importance.csv")
print("\nOpen the reports\\figures folder in VS Code to view the charts.")
