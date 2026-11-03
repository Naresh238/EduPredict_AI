from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "uci_student_performance" / "extracted"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("EDUPREDICT AI - STUDENT RISK CLASSIFICATION")
print("=" * 75)

# 1. Load dataset
df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
df = df.drop_duplicates().copy()

# Predictors do not include G1, G2, or G3.
# Risk labels are derived from G3 for supervised training.
X = df.drop(columns=["G1", "G2", "G3"])
y = (df["G3"] < 10).astype(int)

print("\n1. DATASET")
print("Total students:", len(df))
print("Features:", X.shape[1])
print("At-risk students:", int(y.sum()))
print("Not-at-risk students:", int((y == 0).sum()))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

def make_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

# 2. Define classification models
models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

results = []
fitted_models = {}

# 3. Train and evaluate
print("\n2. MODEL TRAINING AND CROSS-VALIDATION\n")

for name, estimator in models.items():
    pipeline = make_pipeline(estimator)

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    else:
        auc = float("nan")

    row = {
        "Model": name,
        "CV_Accuracy": cv_results["test_accuracy"].mean(),
        "CV_Precision": cv_results["test_precision"].mean(),
        "CV_Recall": cv_results["test_recall"].mean(),
        "CV_F1": cv_results["test_f1"].mean(),
        "CV_ROC_AUC": cv_results["test_roc_auc"].mean(),
        "Test_Accuracy": accuracy_score(y_test, predictions),
        "Test_Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Test_Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Test_F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Test_ROC_AUC": auc,
    }

    results.append(row)
    fitted_models[name] = pipeline

    print(
        f"{name:25s} "
        f"CV F1={row['CV_F1']:.3f} | "
        f"Test F1={row['Test_F1']:.3f} | "
        f"Test Recall={row['Test_Recall']:.3f}"
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "CV_F1", ascending=False
).reset_index(drop=True)

results_df.to_csv(
    REPORT_DIR / "uci_risk_model_comparison.csv",
    index=False,
)

print("\n3. MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 4. Select using cross-validation F1, excluding the baseline.
non_baseline = results_df[
    results_df["Model"] != "Dummy Baseline"
]

best_name = non_baseline.iloc[0]["Model"]
best_model = fitted_models[best_name]

# Save selected model.
model_file = MODEL_DIR / "uci_best_risk_model.joblib"
joblib.dump(best_model, model_file)

# 5. Evaluate selected model on held-out test data.
predictions = best_model.predict(X_test)

if hasattr(best_model, "predict_proba"):
    probabilities = best_model.predict_proba(X_test)[:, 1]
else:
    probabilities = np.full(len(X_test), np.nan)

print("\n4. SELECTED MODEL:", best_name)
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Not At Risk", "At Risk"],
        zero_division=0,
    )
)

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual_Not_At_Risk", "Actual_At_Risk"],
    columns=["Predicted_Not_At_Risk", "Predicted_At_Risk"],
)
matrix_df.to_csv(REPORT_DIR / "uci_risk_confusion_matrix.csv")

print("Confusion matrix:")
print(matrix_df.to_string())

# 6. Save individual test predictions.
prediction_df = X_test.reset_index(drop=True).copy()
prediction_df["Actual_Risk"] = np.where(
    y_test.reset_index(drop=True) == 1,
    "At Risk",
    "Not At Risk",
)
prediction_df["Predicted_Risk"] = np.where(
    predictions == 1,
    "At Risk",
    "Not At Risk",
)
prediction_df["At_Risk_Probability"] = np.round(probabilities, 4)

prediction_df.to_csv(
    REPORT_DIR / "uci_risk_test_predictions.csv",
    index=False,
)

# 7. Save a JSON summary.
best_row = results_df[
    results_df["Model"] == best_name
].iloc[0]

summary = {
    "dataset": "UCI Student Performance - Mathematics",
    "total_students": int(len(df)),
    "training_students": int(len(X_train)),
    "testing_students": int(len(X_test)),
    "target": "Risk derived from G3",
    "risk_definition": "G3 < 10",
    "features_excluded": ["G1", "G2", "G3"],
    "selection_metric": "5-fold cross-validation F1",
    "selected_model": best_name,
    "cv_f1": float(best_row["CV_F1"]),
    "test_accuracy": float(best_row["Test_Accuracy"]),
    "test_precision": float(best_row["Test_Precision"]),
    "test_recall": float(best_row["Test_Recall"]),
    "test_f1": float(best_row["Test_F1"]),
    "test_roc_auc": float(best_row["Test_ROC_AUC"]),
}

summary_file = REPORT_DIR / "uci_best_risk_model_summary.json"
summary_file.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8",
)

print("\n5. FILES SAVED")
print(REPORT_DIR / "uci_risk_model_comparison.csv")
print(REPORT_DIR / "uci_risk_confusion_matrix.csv")
print(REPORT_DIR / "uci_risk_test_predictions.csv")
print(summary_file)
print(model_file)

print("\nSTUDENT RISK CLASSIFICATION COMPLETED SUCCESSFULLY!")
