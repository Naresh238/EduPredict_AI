from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "uci_student_performance" / "extracted"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "student-mat.csv"

print("=" * 75)
print("EDUPREDICT AI - GRADE MODEL HYPERPARAMETER TUNING")
print("=" * 75)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

# --------------------------------------------------
# 2. LOAD AND PREPARE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_FILE, sep=";")
df = df.drop_duplicates().copy()

X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

risk = (y < 10).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

print("\nDATASET INFORMATION")
print("Total students:", len(df))
print("Training students:", len(X_train))
print("Testing students:", len(X_test))
print("Input features:", X.shape[1])
print("Target: G3 - Final mathematics grade")
print("G1 and G2 excluded to avoid earlier-grade leakage.")

# --------------------------------------------------
# 3. DEFINE SEARCH SPACE
# --------------------------------------------------

parameter_space = {
    "model__n_estimators": [200, 300, 400, 500],
    "model__max_depth": [None, 5, 10, 15, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.7, 1.0],
}

candidate_models = {
    "Random Forest": RandomForestRegressor(
        random_state=42,
        n_jobs=1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        random_state=42,
        n_jobs=1,
    ),
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# --------------------------------------------------
# 4. HYPERPARAMETER OPTIMIZATION
# --------------------------------------------------

tuning_results = []
best_search = None
best_model_name = None
best_cv_mae = float("inf")

print("\nSTARTING RANDOMIZED SEARCH")
print("Candidates: Random Forest and Extra Trees")
print("Randomized combinations per model: 20")
print("Cross-validation folds: 5")
print("Scoring metric: MAE")
print("Please wait...\n")

for model_name, estimator in candidate_models.items():

    print("-" * 65)
    print("Tuning:", model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    cv_mae = -search.best_score_

    predictions = search.best_estimator_.predict(X_test)

    test_mae = mean_absolute_error(y_test, predictions)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    test_r2 = r2_score(y_test, predictions)

    result = {
        "Model": model_name,
        "Best_CV_MAE": float(cv_mae),
        "Test_MAE": float(test_mae),
        "Test_RMSE": float(test_rmse),
        "Test_R2": float(test_r2),
        "Best_Parameters": json.dumps(search.best_params_),
    }

    tuning_results.append(result)

    print("Best CV MAE:", round(cv_mae, 4))
    print("Test MAE:", round(test_mae, 4))
    print("Test RMSE:", round(test_rmse, 4))
    print("Test R2:", round(test_r2, 4))
    print("Best parameters:", search.best_params_)

    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_search = search
        best_model_name = model_name

# --------------------------------------------------
# 5. COMPARE TUNED MODELS
# --------------------------------------------------

results_df = pd.DataFrame(tuning_results)
results_df = results_df.sort_values(
    "Best_CV_MAE"
).reset_index(drop=True)

results_file = REPORT_DIR / "uci_grade_tuning_results.csv"
results_df.to_csv(results_file, index=False)

print("\nTUNED MODEL COMPARISON")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 6. SAVE THE BEST TUNED MODEL
# --------------------------------------------------

if best_search is None:
    raise RuntimeError("Hyperparameter tuning did not produce a model.")

best_model = best_search.best_estimator_

model_file = MODEL_DIR / "uci_tuned_grade_model.joblib"
joblib.dump(best_model, model_file)

# --------------------------------------------------
# 7. FINAL TEST-SET EVALUATION
# --------------------------------------------------

best_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, best_predictions)
test_rmse = float(np.sqrt(mean_squared_error(y_test, best_predictions)))
test_r2 = r2_score(y_test, best_predictions)

predictions_df = X_test.reset_index(drop=True).copy()
predictions_df["Actual_G3"] = y_test.reset_index(drop=True)
predictions_df["Predicted_G3"] = np.clip(
    best_predictions, 0, 20
).round(2)
predictions_df["Absolute_Error"] = (
    predictions_df["Actual_G3"] -
    predictions_df["Predicted_G3"]
).abs().round(2)

predictions_file = REPORT_DIR / "uci_tuned_grade_test_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)

# --------------------------------------------------
# 8. SAVE BEST MODEL SUMMARY
# --------------------------------------------------

summary = {
    "project": "EduPredict AI",
    "dataset": "UCI Student Performance - Mathematics",
    "source_file": "student-mat.csv",
    "total_students": int(len(df)),
    "training_students": int(len(X_train)),
    "testing_students": int(len(X_test)),
    "number_of_input_features": int(X.shape[1]),
    "target": "G3",
    "target_range": "0-20",
    "excluded_features": ["G1", "G2"],
    "optimization_method": "RandomizedSearchCV",
    "models_optimized": ["Random Forest", "Extra Trees"],
    "iterations_per_model": 20,
    "cross_validation": "5-fold KFold",
    "selection_metric": "Cross-validation MAE",
    "selected_model": best_model_name,
    "best_parameters": best_search.best_params_,
    "best_cv_mae": float(best_cv_mae),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
    "model_file": "models/uci_tuned_grade_model.joblib",
}

summary_file = REPORT_DIR / "uci_best_grade_tuning_summary.json"
summary_file.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8",
)

# --------------------------------------------------
# 9. CREATE HUMAN-READABLE REPORT
# --------------------------------------------------

report_lines = [
    "EDUPREDICT AI - GRADE MODEL HYPERPARAMETER TUNING REPORT",
    "=" * 65,
    "",
    f"Dataset: {summary['dataset']}",
    f"Total students: {len(df)}",
    f"Training students: {len(X_train)}",
    f"Testing students: {len(X_test)}",
    f"Input features: {X.shape[1]}",
    "Target: G3 - Final mathematics grade (0-20)",
    "Excluded features: G1 and G2",
    "",
    "OPTIMIZATION DETAILS",
    "-" * 40,
    "Method: RandomizedSearchCV",
    "Models: Random Forest and Extra Trees",
    "Iterations per model: 20",
    "Cross-validation: 5-fold KFold",
    "Selection metric: Mean Absolute Error",
    "",
    "SELECTED MODEL",
    "-" * 40,
    f"Model: {best_model_name}",
    f"Best cross-validation MAE: {best_cv_mae:.4f}",
    f"Test MAE: {test_mae:.4f}",
    f"Test RMSE: {test_rmse:.4f}",
    f"Test R2: {test_r2:.4f}",
    "",
    "BEST HYPERPARAMETERS",
    "-" * 40,
    json.dumps(best_search.best_params_, indent=4),
    "",
    "INTERPRETATION",
    "-" * 40,
    "MAE measures average absolute prediction error in grade points.",
    "RMSE gives greater weight to larger prediction errors.",
    "R2 measures the proportion of target variation explained by the model.",
    "Lower MAE and RMSE are better. Higher R2 is generally better.",
    "The test set was kept separate from hyperparameter optimization.",
    "Feature importance does not establish causation.",
    "Performance on this dataset does not guarantee performance in other schools.",
]

report_file = REPORT_DIR / "uci_grade_tuning_report.txt"
report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 75)
print("GRADE MODEL HYPERPARAMETER TUNING COMPLETED")
print("=" * 75)

print("Selected model:", best_model_name)
print(f"Best CV MAE: {best_cv_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2: {test_r2:.4f}")

print("\nFILES CREATED:")
print(model_file)
print(results_file)
print(summary_file)
print(report_file)
print(predictions_file)

print("\nExisting grade model and reports were not overwritten.")
