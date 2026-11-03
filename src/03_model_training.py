from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# EDUPREDICT AI - MODEL TRAINING AND COMPARISON
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EDUPREDICT AI - MODEL TRAINING AND COMPARISON")
print("=" * 70)

# 1. Load the saved training and testing splits
X_train = pd.read_csv(DATA_DIR / "X_train_raw.csv")
X_test = pd.read_csv(DATA_DIR / "X_test_raw.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

print("\n1. DATA LOADED")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

# 2. Identify feature types
numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\n2. NUMERIC FEATURES")
print(numeric_features)

print("\n3. CATEGORICAL FEATURES")
print(categorical_features)

# 3. Create preprocessing pipelines
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# 4. Define candidate models
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

# 5. Cross-validation on training data only
# All imputation, scaling and encoding happen inside each fold.
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

results = []
fitted_models = {}

print("\n4. FIVE-FOLD CROSS-VALIDATION")
print("This may take a few minutes...\n")

for name, estimator in models.items():
    print(f"Evaluating: {name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )

    mean_cv_mae = -cv_results["test_mae"].mean()
    mean_cv_rmse = -cv_results["test_rmse"].mean()
    mean_cv_r2 = cv_results["test_r2"].mean()

    # Fit this candidate on the complete training set.
    pipeline.fit(X_train, y_train)

    # Evaluate against the untouched test set.
    predictions = pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    test_r2 = r2_score(y_test, predictions)

    results.append(
        {
            "Model": name,
            "CV_MAE": mean_cv_mae,
            "CV_RMSE": mean_cv_rmse,
            "CV_R2": mean_cv_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse,
            "Test_R2": test_r2,
        }
    )

    fitted_models[name] = pipeline

    print(
        f"  CV MAE: {mean_cv_mae:.3f} | "
        f"CV RMSE: {mean_cv_rmse:.3f} | "
        f"CV R2: {mean_cv_r2:.3f}"
    )

    print(
        f"  Test MAE: {test_mae:.3f} | "
        f"Test RMSE: {test_rmse:.3f} | "
        f"Test R2: {test_r2:.3f}\n"
    )

# 6. Save comparison results
results_df = pd.DataFrame(results)

# MAE is the primary selection metric.
results_df = results_df.sort_values(
    by="CV_MAE",
    ascending=True,
).reset_index(drop=True)

results_file = REPORT_DIR / "model_comparison.csv"
results_df.to_csv(results_file, index=False)

print("=" * 70)
print("5. MODEL COMPARISON RESULTS")
print("=" * 70)
print(results_df.round(4).to_string(index=False))

# 7. Select the model with the lowest cross-validation MAE
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

print("\n6. BEST MODEL BY CROSS-VALIDATION MAE")
print("Selected model:", best_name)
print(f"Cross-validation MAE: {results_df.iloc[0]['CV_MAE']:.4f}")
print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.4f}")
print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"Test R2: {results_df.iloc[0]['Test_R2']:.4f}")

# 8. Save the selected fitted model.
# joblib is used because it preserves the entire sklearn pipeline.
import joblib

model_file = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_file)

# Save model selection information.
summary = {
    "target": "ML Marks",
    "selected_model": best_name,
    "selection_metric": "Lowest five-fold cross-validation MAE",
    "cv_mae": float(results_df.iloc[0]["CV_MAE"]),
    "cv_rmse": float(results_df.iloc[0]["CV_RMSE"]),
    "cv_r2": float(results_df.iloc[0]["CV_R2"]),
    "test_mae": float(results_df.iloc[0]["Test_MAE"]),
    "test_rmse": float(results_df.iloc[0]["Test_RMSE"]),
    "test_r2": float(results_df.iloc[0]["Test_R2"]),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
}

summary_file = REPORT_DIR / "best_model_summary.json"
summary_file.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8",
)

# 9. Save test predictions for inspection
prediction_df = pd.DataFrame(
    {
        "Actual_ML_Marks": y_test.reset_index(drop=True),
        "Predicted_ML_Marks": best_model.predict(X_test),
    }
)

prediction_df["Absolute_Error"] = (
    prediction_df["Actual_ML_Marks"]
    - prediction_df["Predicted_ML_Marks"]
).abs()

prediction_file = REPORT_DIR / "test_predictions.csv"
prediction_df.to_csv(prediction_file, index=False)

print("\n7. FILES SAVED")
print(results_file)
print(model_file)
print(summary_file)
print(prediction_file)

print("\nMODEL TRAINING AND COMPARISON COMPLETED SUCCESSFULLY!")
