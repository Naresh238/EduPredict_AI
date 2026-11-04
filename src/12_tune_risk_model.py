from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
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
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    ROOT / "data" / "external" / "uci_student_performance"
    / "extracted" / "student-mat.csv"
)
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TARGET_NAME = "At_Risk"


def main():
    print("=" * 75)
    print("EDUPREDICT AI - RISK MODEL HYPERPARAMETER TUNING")
    print("=" * 75)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop_duplicates().copy()

    if "G3" not in df.columns:
        raise ValueError("Dataset must contain the G3 final-grade column.")

    # At-risk means final grade below 10 out of 20.
    y = (df["G3"] < 10).astype(int)

    # Exclude G1, G2 and G3 to prevent target leakage.
    X = df.drop(columns=["G1", "G2", "G3"], errors="ignore")

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

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
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nDATASET INFORMATION")
    print(f"Total students: {len(df)}")
    print(f"Training students: {len(X_train)}")
    print(f"Testing students: {len(X_test)}")
    print(f"Input features: {X.shape[1]}")
    print(f"At-risk students: {int(y.sum())}")
    print(f"Not-at-risk students: {int((y == 0).sum())}")
    print("At-risk definition: G3 < 10")
    print("G1, G2 and G3 excluded from input features.")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    parameter_distributions = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.7, 1.0],
        "model__class_weight": ["balanced", "balanced_subsample", None],
    }

    candidates = {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    best_overall = None
    best_overall_f1 = -1

    print("\nSTARTING RANDOMIZED SEARCH")
    print("Candidates: Random Forest and Extra Trees")
    print("Randomized combinations per model: 20")
    print("Cross-validation folds: 5")
    print("Scoring metric: F1")
    print("Please wait...")

    for model_name, classifier in candidates.items():
        print("\n" + "-" * 65)
        print(f"Tuning: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", classifier),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_distributions,
            n_iter=20,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
            verbose=0,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        roc_auc = roc_auc_score(y_test, probabilities)

        result = {
            "Model": model_name,
            "Best_CV_F1": float(search.best_score_),
            "Test_Accuracy": float(accuracy),
            "Test_Precision": float(precision),
            "Test_Recall": float(recall),
            "Test_F1": float(f1),
            "Test_ROC_AUC": float(roc_auc),
            "Best_Parameters": json.dumps(search.best_params_),
        }
        results.append(result)

        print(f"Best CV F1: {search.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1: {f1:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print(f"Best parameters: {search.best_params_}")

        if search.best_score_ > best_overall_f1:
            best_overall_f1 = search.best_score_
            best_overall = {
                "model_name": model_name,
                "model": best_model,
                "search": search,
                "predictions": predictions,
                "probabilities": probabilities,
                "metrics": result,
            }

    results_df = pd.DataFrame(results).sort_values(
        by="Best_CV_F1",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "uci_risk_tuning_results.csv",
        index=False,
    )

    selected = best_overall
    selected_model = selected["model"]
    selected_predictions = selected["predictions"]
    selected_probabilities = selected["probabilities"]
    selected_metrics = selected["metrics"]

    model_path = MODEL_DIR / "uci_tuned_risk_model.joblib"
    joblib.dump(selected_model, model_path)

    summary = {
        "project": "EduPredict AI",
        "task": "At-risk student classification",
        "dataset": "UCI Student Performance - Mathematics",
        "target": TARGET_NAME,
        "risk_definition": "G3 < 10",
        "excluded_features": ["G1", "G2", "G3"],
        "selected_model": selected["model_name"],
        "best_cv_f1": float(selected["search"].best_score_),
        "test_metrics": {
            "accuracy": selected_metrics["Test_Accuracy"],
            "precision": selected_metrics["Test_Precision"],
            "recall": selected_metrics["Test_Recall"],
            "f1": selected_metrics["Test_F1"],
            "roc_auc": selected_metrics["Test_ROC_AUC"],
        },
        "best_parameters": selected["search"].best_params_,
    }

    with open(
        REPORT_DIR / "uci_best_risk_tuning_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    report = classification_report(
        y_test,
        selected_predictions,
        target_names=["Not At Risk", "At Risk"],
        zero_division=0,
    )

    matrix = confusion_matrix(y_test, selected_predictions)
    pd.DataFrame(
        matrix,
        index=["Actual_Not_At_Risk", "Actual_At_Risk"],
        columns=["Predicted_Not_At_Risk", "Predicted_At_Risk"],
    ).to_csv(REPORT_DIR / "uci_tuned_risk_confusion_matrix.csv")

    test_predictions = pd.DataFrame(
        {
            "Actual_At_Risk": y_test.to_numpy(),
            "Predicted_At_Risk": selected_predictions,
            "At_Risk_Probability": selected_probabilities,
        }
    )
    test_predictions.to_csv(
        REPORT_DIR / "uci_tuned_risk_test_predictions.csv",
        index=False,
    )

    text_report = f"""
EDUPREDICT AI - RISK MODEL HYPERPARAMETER TUNING REPORT

Dataset: UCI Student Performance - Mathematics
Total students: {len(df)}
Training students: {len(X_train)}
Testing students: {len(X_test)}
Input features: {X.shape[1]}

Target: At_Risk
Definition: G3 < 10
G1, G2 and G3 excluded from input features.

Cross-validation: Stratified 5-Fold
Optimization: RandomizedSearchCV
Scoring: F1
Search iterations per candidate: 20

Selected model: {selected['model_name']}

Best CV F1: {selected['search'].best_score_:.4f}
Test Accuracy: {selected_metrics['Test_Accuracy']:.4f}
Test Precision: {selected_metrics['Test_Precision']:.4f}
Test Recall: {selected_metrics['Test_Recall']:.4f}
Test F1: {selected_metrics['Test_F1']:.4f}
Test ROC-AUC: {selected_metrics['Test_ROC_AUC']:.4f}

Best parameters:
{selected['search'].best_params_}

Classification report:
{report}

Confusion matrix:
{matrix}

Note:
These results measure predictive performance, not causal relationships.
The model should support, not replace, human academic intervention.
"""

    (REPORT_DIR / "uci_risk_tuning_report.txt").write_text(
        text_report,
        encoding="utf-8",
    )

    print("\nTUNED MODEL COMPARISON")
    print(results_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("RISK MODEL HYPERPARAMETER TUNING COMPLETED")
    print("=" * 75)
    print(f"Selected model: {selected['model_name']}")
    print(f"Best CV F1: {selected['search'].best_score_:.4f}")
    print(f"Test Accuracy: {selected_metrics['Test_Accuracy']:.4f}")
    print(f"Test Precision: {selected_metrics['Test_Precision']:.4f}")
    print(f"Test Recall: {selected_metrics['Test_Recall']:.4f}")
    print(f"Test F1: {selected_metrics['Test_F1']:.4f}")
    print(f"Test ROC-AUC: {selected_metrics['Test_ROC_AUC']:.4f}")

    print("\nFILES CREATED:")
    print(model_path)
    print(REPORT_DIR / "uci_risk_tuning_results.csv")
    print(REPORT_DIR / "uci_best_risk_tuning_summary.json")
    print(REPORT_DIR / "uci_risk_tuning_report.txt")
    print(REPORT_DIR / "uci_tuned_risk_confusion_matrix.csv")
    print(REPORT_DIR / "uci_tuned_risk_test_predictions.csv")

    print("\nExisting risk model and reports were not overwritten.")


if __name__ == "__main__":
    main()
