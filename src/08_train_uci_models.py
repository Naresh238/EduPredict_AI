from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
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

DATA_FILE = DATA_DIR / "student-mat.csv"

print("=" * 75)
print("EDUPREDICT AI - MACHINE LEARNING MODEL COMPARISON")
print("=" * 75)

# 1. Load the UCI Mathematics dataset
df = pd.read_csv(DATA_FILE, sep=";")
df = df.drop_duplicates().copy()

# G3 is the final mathematics grade, ranging from 0 to 20.
# G1 and G2 are excluded to avoid using earlier grades as predictors.
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

# Create a separate risk label for the later classification task.
risk = (y < 10).astype(int)

X_train, X_test, y_train, y_test, risk_train, risk_test = train_test_split(
    X,
    y,
    risk,
    test_size=0.20,
    random_state=42,
    stratify=risk,
)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

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

print("\n1. DATASET")
print("Total students:", len(df))
print("Training students:", len(X_train))
print("Testing students:", len(X_test))
print("Numeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Target: G3 (final mathematics grade)")
print("G1 and G2 excluded: Yes")

# 2. Define regression models
models = {
    "Dummy Baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
    "Histogram Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    ),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

results = []
fitted_models = {}

# 3. Cross-validation and test evaluation
print("\n2. TRAINING AND EVALUATION")
print("Please wait while the models are evaluated...\n")

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    cv_result = cross_validate(
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

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)

    row = {
        "Model": name,
        "CV_MAE": -cv_result["test_mae"].mean(),
        "CV_RMSE": -cv_result["test_rmse"].mean(),
        "CV_R2": cv_result["test_r2"].mean(),
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
    }

    results.append(row)
    fitted_models[name] = pipeline

    print(
        f"{name:30s} "
        f"CV MAE={row['CV_MAE']:.3f} | "
        f"Test MAE={mae:.3f} | "
        f"Test R2={r2:.3f}"
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("CV_MAE").reset_index(drop=True)

results_file = REPORT_DIR / "uci_model_comparison.csv"
results_df.to_csv(results_file, index=False)

print("\n3. MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# 4. Select the best non-baseline model by cross-validation MAE.
non_baseline = results_df[
    results_df["Model"] != "Dummy Baseline"
].copy()

best_name = non_baseline.iloc[0]["Model"]
best_model = fitted_models[best_name]

# Save the selected model.
import joblib

model_file = MODEL_DIR / "uci_best_grade_model.joblib"
joblib.dump(best_model, model_file)

# 5. Save predictions for further analysis.
best_predictions = best_model.predict(X_test)

predictions_df = X_test.reset_index(drop=True).copy()
predictions_df["Actual_G3"] = y_test.reset_index(drop=True)
predictions_df["Predicted_G3"] = np.clip(
    best_predictions, 0, 20
).round(2)
predictions_df["Absolute_Error"] = (
    predictions_df["Actual_G3"] - predictions_df["Predicted_G3"]
).abs().round(2)
predictions_df["Actual_Risk"] = np.where(
    predictions_df["Actual_G3"] < 10, "At Risk", "Not At Risk"
)
predictions_df.to_csv(
    REPORT_DIR / "uci_test_predictions.csv",
    index=False,
)

# 6. Save a machine-readable summary.
best_row = results_df.loc[results_df["Model"] == best_name].iloc[0]

summary = {
    "dataset": "UCI Student Performance - Mathematics",
    "source_file": "student-mat.csv",
    "total_rows": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "target": "G3",
    "target_range": "0-20",
    "excluded_columns": ["G1", "G2"],
    "selection_metric": "5-fold cross-validation MAE",
    "selected_model": best_name,
    "cv_mae": float(best_row["CV_MAE"]),
    "cv_rmse": float(best_row["CV_RMSE"]),
    "cv_r2": float(best_row["CV_R2"]),
    "test_mae": float(best_row["Test_MAE"]),
    "test_rmse": float(best_row["Test_RMSE"]),
    "test_r2": float(best_row["Test_R2"]),
    "risk_definition": "G3 < 10",
}

summary_file = REPORT_DIR / "uci_best_model_summary.json"
summary_file.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8",
)

# 7. Explain how to interpret the results.
baseline_row = results_df[
    results_df["Model"] == "Dummy Baseline"
].iloc[0]

report_lines = [
    "EDUPREDICT AI - UCI MODEL TRAINING REPORT",
    "=" * 55,
    f"Dataset: {summary['dataset']}",
    f"Total students: {len(df)}",
    f"Training students: {len(X_train)}",
    f"Testing students: {len(X_test)}",
    "Target: G3 (final mathematics grade, 0-20)",
    "G1 and G2 excluded from predictors.",
    "Evaluation: 5-fold cross-validation and held-out test set.",
    "",
    f"Selected model: {best_name}",
    f"Selected model test MAE: {best_row['Test_MAE']:.4f}",
    f"Selected model test RMSE: {best_row['Test_RMSE']:.4f}",
    f"Selected model test R2: {best_row['Test_R2']:.4f}",
    "",
    f"Baseline test MAE: {baseline_row['Test_MAE']:.4f}",
    f"Baseline test RMSE: {baseline_row['Test_RMSE']:.4f}",
    f"Baseline test R2: {baseline_row['Test_R2']:.4f}",
    "",
    "Interpretation:",
    "MAE is the average absolute prediction error in grade points.",
    "Lower MAE and RMSE are better.",
    "R2 can be negative when predictions are worse than predicting",
    "the test-set mean. Compare the selected model with the baseline.",
    "Results are estimates from this dataset, not a guarantee for new schools.",
]

report_file = REPORT_DIR / "uci_model_training_report.txt"
report_file.write_text("\n".join(report_lines), encoding="utf-8")

print("\n4. SELECTED MODEL")
print("Selected model:", best_name)
print(f"Test MAE: {best_row['Test_MAE']:.4f}")
print(f"Test RMSE: {best_row['Test_RMSE']:.4f}")
print(f"Test R2: {best_row['Test_R2']:.4f}")

print("\n5. FILES SAVED")
print(results_file)
print(model_file)
print(REPORT_DIR / "uci_test_predictions.csv")
print(summary_file)
print(report_file)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
