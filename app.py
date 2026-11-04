
import json
import logging
import math
import os
from pathlib import Path

import joblib
import mysql.connector
import pandas as pd
from flask import Flask, abort, render_template, request
from plotly.utils import PlotlyJSONEncoder


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(ROOT / "app" / "templates"),
    static_folder=str(ROOT / "app" / "static"),
)

app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)

app.logger.setLevel(
    os.getenv("LOG_LEVEL", "INFO").upper()
)

REPORT_DIR = ROOT / "reports"

GRADE_MODEL_PATH = (
    ROOT / "models" / "uci_tuned_grade_model.joblib"
)

RISK_MODEL_PATH = (
    ROOT / "models" / "uci_tuned_risk_model.joblib"
)

DATASET_PATH = (
    ROOT
    / "data"
    / "external"
    / "uci_student_performance"
    / "extracted"
    / "student-mat.csv"
)


# ============================================================
# 2. DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("EDUPREDICT_DB_HOST", "localhost"),
    "port": int(os.getenv("EDUPREDICT_DB_PORT", "3306")),
    "user": os.getenv("EDUPREDICT_DB_USER", "edupredict_app"),
    "database": os.getenv(
        "EDUPREDICT_DB_NAME",
        "edupredict_db",
    ),
    "charset": "utf8mb4",
    "connection_timeout": 5,
}


def get_db_connection():
    """Connect to MySQL only when a database operation is needed."""

    password = os.getenv("EDUPREDICT_DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "EDUPREDICT_DB_PASSWORD is not configured."
        )

    config = DB_CONFIG.copy()
    config["password"] = password

    return mysql.connector.connect(**config)


def ensure_predictions_table(connection):
    """Create the prediction-history table when necessary."""

    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,
                predicted_grade DOUBLE NOT NULL,
                risk_status VARCHAR(50) NOT NULL,
                risk_probability DOUBLE NULL,
                student_details JSON NULL
            ) ENGINE=InnoDB
              DEFAULT CHARACTER SET utf8mb4
              COLLATE utf8mb4_unicode_ci
            """
        )

        connection.commit()

    finally:
        cursor.close()


def init_database():
    """Initialize the database manually when required."""

    connection = None

    try:
        connection = get_db_connection()
        ensure_predictions_table(connection)

        app.logger.info(
            "EduPredict prediction-history table is ready."
        )

    finally:
        if connection is not None:
            connection.close()


# ============================================================
# 3. LOAD TRAINED MODELS AND DATASET
# ============================================================

def load_required_model(path):
    """Load a trained model and report a clear error if unavailable."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required model file was not found: {path}"
        )

    try:
        model = joblib.load(path)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load model: {path.name}"
        ) from exc

    if not hasattr(model, "predict"):
        raise TypeError(
            f"{path.name} is not a valid prediction model."
        )

    return model


if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Student dataset not found: {DATASET_PATH}"
    )

dataset = pd.read_csv(
    DATASET_PATH,
    sep=";",
)

if dataset.empty:
    raise ValueError(
        "The student dataset is empty."
    )

required_columns = {"G1", "G2", "G3"}

if not required_columns.issubset(dataset.columns):
    raise ValueError(
        "The student dataset must contain G1, G2 and G3."
    )

grade_model = load_required_model(
    GRADE_MODEL_PATH
)

risk_model = load_required_model(
    RISK_MODEL_PATH
)


# Both models use the student input features without G1, G2 or G3.
RISK_FEATURES = [
    column
    for column in dataset.columns
    if column not in ["G1", "G2", "G3"]
]

GRADE_FEATURES = RISK_FEATURES.copy()


# Check that the loaded models expect the same features.
for model_name, model in [
    ("Grade", grade_model),
    ("Risk", risk_model),
]:
    model_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_features is not None:
        if list(model_features) != RISK_FEATURES:
            raise ValueError(
                f"{model_name} model feature order does not "
                "match the dataset feature order."
            )


# ============================================================
# 4. INPUT FIELD METADATA
# ============================================================

def get_feature_metadata():
    """Generate form fields using the training dataset."""

    metadata = []

    for column in RISK_FEATURES:

        if pd.api.types.is_numeric_dtype(
            dataset[column]
        ):

            metadata.append(
                {
                    "name": column,
                    "type": "number",
                    "minimum": float(
                        dataset[column].min()
                    ),
                    "maximum": float(
                        dataset[column].max()
                    ),
                    "default": float(
                        dataset[column].median()
                    ),
                    "choices": [],
                }
            )

        else:

            choices = sorted(
                dataset[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            metadata.append(
                {
                    "name": column,
                    "type": "select",
                    "minimum": None,
                    "maximum": None,
                    "default": (
                        choices[0]
                        if choices
                        else ""
                    ),
                    "choices": choices,
                }
            )

    return metadata


FEATURE_METADATA = get_feature_metadata()


# ============================================================
# 5. STUDENT INPUT VALIDATION
# ============================================================

def create_input_dataframe(form_data):
    """
    Validate submitted student information.

    Checks:
    - Required fields
    - Numeric values
    - NaN and infinity
    - Dataset minimum and maximum
    - Whole-number fields
    - Valid categorical choices
    """

    values = {}

    for feature in RISK_FEATURES:

        raw_value = form_data.get(
            feature,
            "",
        )

        raw_value = str(raw_value).strip()

        if raw_value == "":
            raise ValueError(
                f"Please provide a value for {feature}."
            )

        if pd.api.types.is_numeric_dtype(
            dataset[feature]
        ):

            try:
                value = float(raw_value)

            except (TypeError, ValueError):
                raise ValueError(
                    f"{feature} must be a valid number."
                ) from None

            if not math.isfinite(value):
                raise ValueError(
                    f"{feature} must be a finite number."
                )

            minimum = float(
                dataset[feature].min()
            )

            maximum = float(
                dataset[feature].max()
            )

            if value < minimum or value > maximum:
                raise ValueError(
                    f"{feature} must be between "
                    f"{minimum:g} and {maximum:g}."
                )

            if pd.api.types.is_integer_dtype(
                dataset[feature]
            ):

                if not value.is_integer():
                    raise ValueError(
                        f"{feature} must be a whole number."
                    )

                value = int(value)

            values[feature] = value

        else:

            valid_choices = (
                dataset[feature]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if raw_value not in valid_choices:
                raise ValueError(
                    f"Invalid choice for {feature}."
                )

            values[feature] = raw_value

    return pd.DataFrame(
        [values],
        columns=RISK_FEATURES,
    )


# ============================================================
# 6. REPORT FILE LOADING
# ============================================================

def load_json_report(filename):
    """Load a JSON report without crashing the home page."""

    path = REPORT_DIR / filename

    if not path.exists():
        app.logger.warning(
            "Report file not found: %s",
            path,
        )
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        app.logger.exception(
            "Could not read report: %s",
            path,
        )
        return {}


def load_csv_report(filename):
    """Load an analytics CSV report with validation."""

    path = REPORT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Analytics report is missing: {filename}"
        )

    report = pd.read_csv(path)

    if report.empty:
        raise ValueError(
            f"Analytics report is empty: {filename}"
        )

    return report


# ============================================================
# 7. SAVE PREDICTION HISTORY
# ============================================================

def save_prediction(result, form_values):
    """Save a prediction to MySQL."""

    connection = None
    cursor = None

    try:
        connection = get_db_connection()

        ensure_predictions_table(connection)

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (
                predicted_grade,
                risk_status,
                risk_probability,
                student_details
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                result["grade"],
                result["risk"],
                result["risk_probability"],
                json.dumps(
                    form_values,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            ),
        )

        connection.commit()

    except Exception:
        if connection is not None:
            connection.rollback()

        raise

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# 8. HOME PAGE
# ============================================================

@app.route("/")
def home():

    grade_summary = load_json_report(
        "uci_best_model_summary.json"
    )

    risk_summary = load_json_report(
        "uci_best_risk_model_summary.json"
    )

    return render_template(
        "index.html",
        total_students=len(dataset),
        grade_model=grade_summary,
        risk_model=risk_summary,
    )


# ============================================================
# 9. STUDENT PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"],
)
def predict():

    result = None
    error = None
    warning = None
    form_values = {}

    if request.method == "POST":

        form_values = request.form.to_dict()

        try:

            # Step 1: Validate submitted student information.
            student_data = create_input_dataframe(
                request.form
            )

            # Step 2: Predict final grade.
            grade_prediction = float(
                grade_model.predict(
                    student_data[GRADE_FEATURES]
                )[0]
            )

            if not math.isfinite(
                grade_prediction
            ):
                raise ValueError(
                    "The grade model returned an invalid prediction."
                )

            # Final grade is on the UCI dataset's 0-20 scale.
            grade_prediction = max(
                0.0,
                min(
                    20.0,
                    grade_prediction,
                ),
            )

            # Step 3: Predict academic risk.
            risk_prediction = int(
                risk_model.predict(
                    student_data[RISK_FEATURES]
                )[0]
            )

            if risk_prediction not in (0, 1):
                raise ValueError(
                    "The risk model returned an invalid class."
                )

            # Step 4: Calculate probability of class 1 (At Risk).
            risk_probability = None

            if hasattr(
                risk_model,
                "predict_proba",
            ):

                classes = list(
                    risk_model.classes_
                )

                if 1 in classes:

                    probabilities = (
                        risk_model.predict_proba(
                            student_data[RISK_FEATURES]
                        )[0]
                    )

                    risk_probability = float(
                        probabilities[
                            classes.index(1)
                        ]
                    )

                    if not math.isfinite(
                        risk_probability
                    ):
                        raise ValueError(
                            "The risk model returned an invalid probability."
                        )

                    risk_probability = max(
                        0.0,
                        min(
                            1.0,
                            risk_probability,
                        ),
                    )

            # Step 5: Prepare the prediction result.
            result = {
                "grade": round(
                    grade_prediction,
                    2,
                ),
                "risk": (
                    "At Risk"
                    if risk_prediction == 1
                    else "Not At Risk"
                ),
                "risk_probability": (
                    round(
                        risk_probability * 100,
                        2,
                    )
                    if risk_probability is not None
                    else None
                ),
            }

            # Step 6: Save history separately.
            # A MySQL failure must not cancel a valid prediction.
            try:

                save_prediction(
                    result,
                    form_values,
                )

            except Exception:

                app.logger.exception(
                    "Prediction generated but history saving failed."
                )

                warning = (
                    "Your prediction was generated successfully, "
                    "but it could not be saved to history. "
                    "Please check the database connection."
                )

        except (
            ValueError,
            TypeError,
        ) as exc:

            result = None
            error = str(exc)

        except Exception:

            app.logger.exception(
                "Student prediction failed."
            )

            result = None

            error = (
                "Prediction could not be completed. "
                "Please check your input and try again."
            )

    return render_template(
        "predict.html",
        features=FEATURE_METADATA,
        result=result,
        error=error,
        warning=warning,
        form_values=form_values,
    )


# ============================================================
# 10. PREDICTION HISTORY
# ============================================================

@app.route("/history")
def history():

    connection = None
    cursor = None

    predictions = []
    error = None

    try:

        connection = get_db_connection()

        ensure_predictions_table(connection)

        cursor = connection.cursor(
            dictionary=True
        )

        cursor.execute(
            """
            SELECT
                id,
                created_at,
                predicted_grade,
                risk_status,
                risk_probability,
                student_details
            FROM predictions
            ORDER BY id DESC
            LIMIT 200
            """
        )

        predictions = cursor.fetchall()

    except Exception:

        app.logger.exception(
            "Could not load prediction history."
        )

        error = (
            "Prediction history is currently unavailable. "
            "Please check the database connection."
        )

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "history.html",
        predictions=predictions,
        total_predictions=len(predictions),
        error=error,
    )


# ============================================================
# 11. ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    database_status = "ok"
    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute("SELECT 1")

        cursor.fetchone()

    except Exception:

        app.logger.exception(
            "Health check: MySQL connection failed."
        )

        database_status = "error"

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    status = (
        "ok"
        if database_status == "ok"
        else "degraded"
    )

    return {
        "status": status,
        "application": "EduPredict AI",
        "grade_model_loaded": grade_model is not None,
        "risk_model_loaded": risk_model is not None,
        "dataset_loaded": not dataset.empty,
        "database": database_status,
    }, (
        200
        if database_status == "ok"
        else 503
    )


# ============================================================
# 13. ANALYTICS DASHBOARD
# ============================================================

@app.route("/analytics")
def analytics():

    import plotly.graph_objects as go

    try:

        # Load model evaluation reports.
        regression = load_csv_report(
            "uci_model_comparison.csv"
        )

        classification = load_csv_report(
            "uci_risk_model_comparison.csv"
        )

        importance = load_csv_report(
            "uci_risk_permutation_importance.csv"
        )

        # Validate required columns.
        required_regression_columns = {
            "Model",
            "CV_MAE",
            "Test_MAE",
        }

        required_classification_columns = {
            "Model",
            "CV_F1",
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
        }

        required_importance_columns = {
            "Feature",
            "Importance_Mean",
            "Importance_Std",
        }

        if not required_regression_columns.issubset(
            regression.columns
        ):
            raise ValueError(
                "Regression report is missing required columns."
            )

        if not required_classification_columns.issubset(
            classification.columns
        ):
            raise ValueError(
                "Classification report is missing required columns."
            )

        if not required_importance_columns.issubset(
            importance.columns
        ):
            raise ValueError(
                "Feature-importance report is missing required columns."
            )

        # Student grade distribution.
        grades = pd.to_numeric(
            dataset["G3"],
            errors="coerce",
        ).dropna()

        if grades.empty:
            raise ValueError(
                "No valid final-grade values were found."
            )

        # This is the project's defined risk label,
        # not a medical or official academic classification.
        risk_labels = grades < 10

        at_risk_count = int(
            risk_labels.sum()
        )

        total_students = int(
            len(grades)
        )

        at_risk_percent = round(
            at_risk_count / total_students * 100,
            1,
        )

        # Select regression model summary.
        grade_row = regression.loc[
            regression["Model"].eq(
                "Random Forest"
            )
        ]

        if grade_row.empty:

            grade_row = regression.sort_values(
                "CV_MAE"
            ).head(1)

        if grade_row.empty:
            raise ValueError(
                "No regression model results are available."
            )

        grade_mae = float(
            grade_row.iloc[0]["Test_MAE"]
        )

        if not math.isfinite(grade_mae):
            raise ValueError(
                "Invalid grade MAE in the model report."
            )

        # Select the best non-baseline risk model by CV F1.
        eligible_models = classification.loc[
            classification["Model"].ne(
                "Dummy Baseline"
            )
        ]

        if eligible_models.empty:
            eligible_models = classification

        eligible_models = eligible_models.dropna(
            subset=["CV_F1"]
        )

        if eligible_models.empty:
            raise ValueError(
                "No valid classification model results are available."
            )

        selected_row = eligible_models.sort_values(
            "CV_F1",
            ascending=False,
        ).iloc[0]

        risk_recall = (
            float(selected_row["Test_Recall"]) * 100
        )

        risk_auc = float(
            selected_row["Test_ROC_AUC"]
        )

        if not (
            math.isfinite(risk_recall)
            and math.isfinite(risk_auc)
        ):
            raise ValueError(
                "Invalid classification metrics in the model report."
            )

        # Prepare Plotly charts.
        charts = {}

        # Chart 1: Risk distribution.
        charts["risk"] = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "At Risk",
                        "Not At Risk",
                    ],
                    values=[
                        at_risk_count,
                        total_students - at_risk_count,
                    ],
                    hole=0.55,
                    textinfo="label+percent",
                    hovertemplate=(
                        "%{label}: %{value} students"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                title="Students by project-defined risk label",
                showlegend=True,
            ),
        ).to_plotly_json()

        # Chart 2: Grade distribution.
        charts["grade"] = go.Figure(
            data=[
                go.Histogram(
                    x=grades.tolist(),
                    xbins={
                        "start": 0,
                        "end": 21,
                        "size": 1,
                    },
                    name="Students",
                    hovertemplate=(
                        "Grade: %{x}<br>"
                        "Students: %{y}"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                xaxis_title="Final grade (G3)",
                yaxis_title="Number of students",
                bargap=0.08,
            ),
        ).to_plotly_json()

        # Chart 3: Regression model comparison.
        regression_sorted = regression.sort_values(
            "Test_MAE"
        )

        charts["regression"] = go.Figure(
            data=[
                go.Bar(
                    x=regression_sorted[
                        "Test_MAE"
                    ].tolist(),
                    y=regression_sorted[
                        "Model"
                    ].tolist(),
                    orientation="h",
                    text=regression_sorted[
                        "Test_MAE"
                    ].round(3).tolist(),
                    textposition="auto",
                    hovertemplate=(
                        "%{y}<br>"
                        "Test MAE: %{x:.3f}"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                xaxis_title="Test MAE (lower is better)",
                yaxis={
                    "autorange": "reversed"
                },
            ),
        ).to_plotly_json()

        # Chart 4: Classification model comparison.
        classification_sorted = classification.sort_values(
            "Test_F1"
        )

        charts["classification"] = go.Figure(
            data=[
                go.Bar(
                    x=classification_sorted[
                        "Test_F1"
                    ].tolist(),
                    y=classification_sorted[
                        "Model"
                    ].tolist(),
                    orientation="h",
                    text=classification_sorted[
                        "Test_F1"
                    ].round(3).tolist(),
                    textposition="auto",
                    hovertemplate=(
                        "%{y}<br>"
                        "Test F1: %{x:.3f}"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                xaxis_title="Test F1-score",
                xaxis={
                    "range": [0, 1]
                },
                yaxis={
                    "autorange": "reversed"
                },
            ),
        ).to_plotly_json()

        # Chart 5: Top ten feature importances.
        top_importance = importance.sort_values(
            "Importance_Mean",
            ascending=True,
        ).tail(10)

        charts["importance"] = go.Figure(
            data=[
                go.Bar(
                    x=top_importance[
                        "Importance_Mean"
                    ].tolist(),
                    y=top_importance[
                        "Feature"
                    ].tolist(),
                    orientation="h",
                    error_x={
                        "type": "data",
                        "array": (
                            top_importance[
                                "Importance_Std"
                            ]
                            .fillna(0)
                            .tolist()
                        ),
                        "visible": True,
                    },
                    hovertemplate=(
                        "%{y}<br>"
                        "Mean decrease in test F1: %{x:.4f}"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                xaxis_title="Mean decrease in test F1",
                yaxis={
                    "autorange": "reversed"
                },
            ),
        ).to_plotly_json()

        # Chart 6: Selected risk model evaluation.
        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]

        metric_columns = [
            "Test_Accuracy",
            "Test_Precision",
            "Test_Recall",
            "Test_F1",
            "Test_ROC_AUC",
        ]

        metric_values = [
            float(selected_row[column])
            for column in metric_columns
        ]

        if not all(
            math.isfinite(value)
            for value in metric_values
        ):
            raise ValueError(
                "Invalid evaluation metrics were found."
            )

        charts["evaluation"] = go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    text=[
                        f"{value:.3f}"
                        for value in metric_values
                    ],
                    textposition="auto",
                    hovertemplate=(
                        "%{x}: %{y:.3f}"
                        "<extra></extra>"
                    ),
                )
            ],
            layout=go.Layout(
                yaxis_title="Score",
                yaxis={
                    "range": [0, 1]
                },
            ),
        ).to_plotly_json()

        charts_json = json.dumps(
            charts,
            cls=PlotlyJSONEncoder,
            allow_nan=False,
        )

        return render_template(
            "analytics.html",
            total_students=total_students,
            at_risk_count=at_risk_count,
            at_risk_percent=at_risk_percent,
            grade_mae=f"{grade_mae:.3f}",
            risk_recall=f"{risk_recall:.1f}",
            risk_auc=f"{risk_auc:.3f}",
            charts_json=charts_json,
        )

    except Exception:

        app.logger.exception(
            "Could not generate the analytics dashboard."
        )

        abort(
            503,
            description=(
                "Analytics reports are unavailable. "
                "Check the reports folder and try again."
            ),
        )


# ============================================================
# 14. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        host=os.getenv(
            "FLASK_HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv("PORT", "5000")
        ),
        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "false",
            ).lower() == "true"
        ),
    )
