from pathlib import Path
import re
import sys

app_path = Path("app.py")
source = app_path.read_text(encoding="utf-8")

if '@app.route("/analytics")' in source or "@app.route('/analytics')" in source:
    print("An analytics route already exists. No changes made.")
    sys.exit(0)

marker = 'if __name__ == "__main__":'
if marker not in source:
    print("ERROR: Could not find the Flask startup block. No changes made.")
    sys.exit(1)

backup_path = Path("app.py.analytics-backup")
backup_path.write_text(source, encoding="utf-8")

route_code = r'''

@app.route("/analytics")
def analytics():
    import plotly.graph_objects as go
    from plotly.utils import PlotlyJSONEncoder

    # Read the existing model evaluation reports.
    regression = pd.read_csv(
        REPORT_DIR / "uci_model_comparison.csv"
    )
    classification = pd.read_csv(
        REPORT_DIR / "uci_risk_model_comparison.csv"
    )
    importance = pd.read_csv(
        REPORT_DIR / "uci_risk_permutation_importance.csv"
    )

    # Calculate dataset-level risk distribution from the original data.
    grades = pd.to_numeric(dataset["G3"], errors="coerce").dropna()
    risk_labels = grades < 10
    at_risk_count = int(risk_labels.sum())
    total_students = int(len(grades))
    at_risk_percent = (
        round(at_risk_count / total_students * 100, 1)
        if total_students else 0
    )

    # Find the selected grade model using the saved model comparison.
    grade_row = regression.loc[
        regression["Model"].eq("Random Forest")
    ]
    if grade_row.empty:
        grade_row = regression.sort_values("CV_MAE").head(1)

    grade_mae = float(grade_row.iloc[0]["Test_MAE"])

    # The risk classifier was selected by cross-validation F1.
    eligible_models = classification[
        classification["Model"].ne("Dummy Baseline")
    ]
    selected_row = eligible_models.sort_values(
        "CV_F1", ascending=False
    ).iloc[0]

    risk_recall = float(selected_row["Test_Recall"]) * 100
    risk_auc = float(selected_row["Test_ROC_AUC"])

    def figure_payload(figure):
        return figure.to_plotly_json()

    charts = {}

    # 1. Student risk distribution.
    charts["risk"] = figure_payload(
        go.Figure(
            data=[
                go.Pie(
                    labels=["At Risk", "Not At Risk"],
                    values=[
                        at_risk_count,
                        total_students - at_risk_count,
                    ],
                    hole=0.55,
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} students<extra></extra>",
                )
            ],
            layout=go.Layout(
                title="Students by project-defined risk label",
                showlegend=True,
            ),
        )
    )

    # 2. Final grade distribution.
    charts["grade"] = figure_payload(
        go.Figure(
            data=[
                go.Histogram(
                    x=grades.tolist(),
                    xbins={"start": 0, "end": 21, "size": 1},
                    name="Students",
                    hovertemplate="Grade: %{x}<br>Students: %{y}<extra></extra>",
                )
            ],
            layout=go.Layout(
                xaxis_title="Final grade (G3)",
                yaxis_title="Number of students",
                bargap=0.08,
            ),
        )
    )

    # 3. Regression model comparison by test MAE.
    regression_sorted = regression.sort_values("Test_MAE")
    charts["regression"] = figure_payload(
        go.Figure(
            data=[
                go.Bar(
                    x=regression_sorted["Test_MAE"].tolist(),
                    y=regression_sorted["Model"].tolist(),
                    orientation="h",
                    text=regression_sorted["Test_MAE"].round(3).tolist(),
                    textposition="auto",
                    hovertemplate="%{y}<br>Test MAE: %{x:.3f}<extra></extra>",
                )
            ],
            layout=go.Layout(
                xaxis_title="Test MAE (lower is better)",
                yaxis={"autorange": "reversed"},
            ),
        )
    )

    # 4. Risk classification model comparison by test F1.
    classification_sorted = classification.sort_values("Test_F1")
    charts["classification"] = figure_payload(
        go.Figure(
            data=[
                go.Bar(
                    x=classification_sorted["Test_F1"].tolist(),
                    y=classification_sorted["Model"].tolist(),
                    orientation="h",
                    text=classification_sorted["Test_F1"].round(3).tolist(),
                    textposition="auto",
                    hovertemplate="%{y}<br>Test F1: %{x:.3f}<extra></extra>",
                )
            ],
            layout=go.Layout(
                xaxis_title="Test F1-score",
                xaxis={"range": [0, 1]},
                yaxis={"autorange": "reversed"},
            ),
        )
    )

    # 5. Top 10 permutation-importance features.
    top_importance = importance.sort_values(
        "Importance_Mean", ascending=True
    ).tail(10)

    charts["importance"] = figure_payload(
        go.Figure(
            data=[
                go.Bar(
                    x=top_importance["Importance_Mean"].tolist(),
                    y=top_importance["Feature"].tolist(),
                    orientation="h",
                    error_x={
                        "type": "data",
                        "array": top_importance["Importance_Std"].fillna(0).tolist(),
                        "visible": True,
                    },
                    hovertemplate="%{y}<br>Mean F1 decrease: %{x:.4f}<extra></extra>",
                )
            ],
            layout=go.Layout(
                xaxis_title="Mean decrease in test F1",
                yaxis={"autorange": "reversed"},
            ),
        )
    )

    # 6. Evaluation metrics for the selected risk classifier.
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    metric_columns = [
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
        "Test_ROC_AUC",
    ]
    metric_values = [
        float(selected_row[column]) for column in metric_columns
    ]

    charts["evaluation"] = figure_payload(
        go.Figure(
            data=[
                go.Bar(
                    x=metric_names,
                    y=metric_values,
                    text=[f"{value:.3f}" for value in metric_values],
                    textposition="auto",
                    hovertemplate="%{x}: %{y:.3f}<extra></extra>",
                )
            ],
            layout=go.Layout(
                yaxis_title="Score",
                yaxis={"range": [0, 1]},
            ),
        )
    )

    charts_json = json.dumps(charts, cls=PlotlyJSONEncoder)

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

'''

source = source.replace(marker, route_code + "\n" + marker, 1)
app_path.write_text(source, encoding="utf-8")
print("SUCCESS: Added /analytics route.")
print(f"Backup created: {backup_path.resolve()}")
