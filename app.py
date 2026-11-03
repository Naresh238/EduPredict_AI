from pathlib import Path
import json
import os
import joblib
import mysql.connector
import pandas as pd
from flask import Flask, render_template, request
from plotly.utils import PlotlyJSONEncoder

ROOT = Path(__file__).resolve().parent
app = Flask(
    __name__,
    template_folder=str(ROOT / "app" / "templates"),
    static_folder=str(ROOT / "app" / "static"),
)
DB_CONFIG = {
    "host": os.getenv("EDUPREDICT_DB_HOST", "localhost"),
    "port": int(os.getenv("EDUPREDICT_DB_PORT", "3306")),
    "user": os.getenv("EDUPREDICT_DB_USER", "edupredict_app"),
    "database": os.getenv("EDUPREDICT_DB_NAME", "edupredict_db"),
    "charset": "utf8mb4",
}
def get_db_connection():
    """Connect to the EduPredict MySQL database."""
    password = os.getenv("EDUPREDICT_DB_PASSWORD")

    if not password:
        raise RuntimeError(
            "EDUPREDICT_DB_PASSWORD is not set. "
            "Set the variable in PowerShell before starting the app."
        )

    config = DB_CONFIG.copy()
    config["password"] = password

    return mysql.connector.connect(**config)
def init_database():
    """Create the prediction-history table if it does not exist."""
    connection = get_db_connection()
    cursor = None
    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
        if cursor is not None:
            cursor.close()

        connection.close()



init_database()
GRADE_MODEL_PATH = ROOT / "models" / "uci_best_grade_model.joblib"
RISK_MODEL_PATH = ROOT / "models" / "uci_best_risk_model.joblib"
REPORT_DIR = ROOT / "reports"

grade_model = joblib.load(GRADE_MODEL_PATH)
risk_model = joblib.load(RISK_MODEL_PATH)

DATASET_PATH = (
    ROOT
    / "data"
    / "external"
    / "uci_student_performance"
    / "extracted"
    / "student-mat.csv"
)

dataset = pd.read_csv(DATASET_PATH, sep=";")

RISK_FEATURES = [
    column
    for column in dataset.columns
    if column not in ["G1", "G2", "G3"]
]

GRADE_FEATURES = [
    column
    for column in dataset.columns
    if column not in ["G1", "G2", "G3"]
]
def get_feature_metadata():
    metadata = []

    for column in RISK_FEATURES:

        if pd.api.types.is_numeric_dtype(dataset[column]):

            metadata.append(
                {
                    "name": column,
                    "type": "number",
                    "minimum": float(dataset[column].min()),
                    "maximum": float(dataset[column].max()),
                    "default": float(dataset[column].median()),
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
                    "default": choices[0] if choices else "",
                    "choices": choices,
                }
            )

    return metadata


FEATURE_METADATA = get_feature_metadata()
def create_input_dataframe(form_data):
    values = {}

    for feature in RISK_FEATURES:

        raw_value = form_data.get(feature, "").strip()

        if raw_value == "":
            raise ValueError(
                f"Please provide a value for {feature}."
            )

        if pd.api.types.is_numeric_dtype(dataset[feature]):

            try:
                value = float(raw_value)

            except (TypeError, ValueError):
                raise ValueError(
                    f"{feature} must be a valid number."
                ) from None

            minimum = float(dataset[feature].min())
            maximum = float(dataset[feature].max())

            if value < minimum or value > maximum:
                raise ValueError(
                    f"{feature} must be between "
                    f"{minimum:g} and {maximum:g}."
                )

            if pd.api.types.is_integer_dtype(dataset[feature]):

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




def load_json_report(filename):
    path = REPORT_DIR / filename

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError):
        app.logger.exception(
            "Could not read report file: %s",
            path,
        )
        return {}
def save_prediction(result, form_values):
    connection = get_db_connection()
    cursor = None

    try:
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
                ),
            ),
        )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()

        connection.close()




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
@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None
    error = None
    form_values = {}

    if request.method == "POST":

        form_values = request.form.to_dict()

        try:

            student_data = create_input_dataframe(
                request.form
            )

           
            grade_prediction = float(
                grade_model.predict(student_data)[0]
            )

            grade_prediction = max(
                0.0,
                min(20.0, grade_prediction),
            )

            
            risk_prediction = int(
                risk_model.predict(student_data)[0]
            )

            risk_probability = None

            if hasattr(risk_model, "predict_proba"):

                classes = list(risk_model.classes_)

                if 1 in classes:

                    risk_probability = float(
                        risk_model.predict_proba(
                            student_data
                        )[0][classes.index(1)]
                    )

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

            
            save_prediction(
                result,
                form_values,
            )

        except (ValueError, TypeError) as exc:

            error = str(exc)

        except Exception:

            app.logger.exception(
                "Prediction or database save failed"
            )

            error = (
                "Prediction could not be completed or saved. "
                "Please check the submitted values and "
                "database connection."
            )

            result = None

    return render_template(
        "predict.html",
        features=FEATURE_METADATA,
        result=result,
        error=error,
        form_values=form_values,
    )




@app.route("/history")
def history():

    connection = get_db_connection()
    cursor = None

    try:

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

    finally:

        if cursor is not None:
            cursor.close()

        connection.close()

    return render_template(
        "history.html",
        predictions=predictions,
        total_predictions=len(predictions),
    )




@app.route("/about")
def about():
    return render_template("about.html")




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
            "Health check: MySQL connection failed"
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
        "database": database_status,
    }, (
        200
        if database_status == "ok"
        else 503
    )




@app.route("/analytics")
def analytics():

    import plotly.graph_objects as go

    
    regression = pd.read_csv(
        REPORT_DIR / "uci_model_comparison.csv"
    )

    classification = pd.read_csv(
        REPORT_DIR / "uci_risk_model_comparison.csv"
    )

    importance = pd.read_csv(
        REPORT_DIR / "uci_risk_permutation_importance.csv"
    )

   
    grades = pd.to_numeric(
        dataset["G3"],
        errors="coerce",
    ).dropna()

    risk_labels = grades < 10

    at_risk_count = int(
        risk_labels.sum()
    )

    total_students = int(
        len(grades)
    )

    at_risk_percent = (
        round(
            at_risk_count / total_students * 100,
            1,
        )
        if total_students
        else 0
    )

   
    grade_row = regression.loc[
        regression["Model"].eq("Random Forest")
    ]

    if grade_row.empty:
        grade_row = regression.sort_values(
            "CV_MAE"
        ).head(1)

    grade_mae = float(
        grade_row.iloc[0]["Test_MAE"]
    )

   
    eligible_models = classification[
        classification["Model"].ne(
            "Dummy Baseline"
        )
    ]

    if eligible_models.empty:
        eligible_models = classification

    if eligible_models.empty:
        raise ValueError(
            "The classification model report contains no models."
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

    charts = {}

    
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
                    "Students: %{y}<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            xaxis_title="Final grade (G3)",
            yaxis_title="Number of students",
            bargap=0.08,
        ),
    ).to_plotly_json()

   
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
                    "Test MAE: %{x:.3f}<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            xaxis_title="Test MAE (lower is better)",
            yaxis={"autorange": "reversed"},
        ),
    ).to_plotly_json()

   
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
                    "Test F1: %{x:.3f}<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            xaxis_title="Test F1-score",
            xaxis={"range": [0, 1]},
            yaxis={"autorange": "reversed"},
        ),
    ).to_plotly_json()

   
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
                    "Mean F1 decrease: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            xaxis_title="Mean decrease in test F1",
            yaxis={"autorange": "reversed"},
        ),
    ).to_plotly_json()

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
                    "%{x}: %{y:.3f}<extra></extra>"
                ),
            )
        ],
        layout=go.Layout(
            yaxis_title="Score",
            yaxis={"range": [0, 1]},
        ),
    ).to_plotly_json()

    charts_json = json.dumps(
        charts,
        cls=PlotlyJSONEncoder,
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
                "true",
            ).lower() == "true"
        ),
    )