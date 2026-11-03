from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "student_performance.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_FILE)

print("=" * 70)
print("EDUPREDICT AI - DATA QUALITY INVESTIGATION")
print("=" * 70)

# 1. Basic dataset information
print("\n1. DATASET INFORMATION")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn data types:")
print(df.dtypes.to_string())

# 2. Target distribution
print("\n2. ML MARKS DISTRIBUTION")
print(df["ML Marks"].describe().round(3).to_string())

print("\nNumber of unique ML Marks values:")
print(df["ML Marks"].nunique())

# 3. Compare Python Marks and ML Marks
print("\n3. PYTHON MARKS VS ML MARKS")

valid_scores = df[["Python Marks", "ML Marks"]].dropna()

if len(valid_scores) > 2:
    correlation = valid_scores["Python Marks"].corr(
        valid_scores["ML Marks"]
    )

    print("Records with both scores:", len(valid_scores))
    print(f"Pearson correlation: {correlation:.4f}")

    print("\nFirst 15 score pairs:")
    print(valid_scores.head(15).to_string(index=False))

# 4. Check whether the student identifier is sequential
print("\n4. STUDENT ID CHECK")

if "Student ID" in df.columns:
    print("Unique Student IDs:", df["Student ID"].nunique())
    print("Missing Student IDs:", df["Student ID"].isna().sum())

# 5. Check whether dates appear sequential
print("\n5. JOIN DATE CHECK")

if "Join Date" in df.columns:
    dates = pd.to_datetime(df["Join Date"], errors="coerce")

    print("Valid dates:", dates.notna().sum())
    print("Missing or invalid dates:", dates.isna().sum())
    print("Earliest date:", dates.min())
    print("Latest date:", dates.max())
    print("Unique dates:", dates.nunique())

# 6. Compare score averages by attendance group
print("\n6. ML MARKS BY ATTENDANCE GROUP")

if "Attendance" in df.columns:
    attendance_data = df[["Attendance", "ML Marks"]].dropna().copy()

    attendance_data["Attendance Group"] = pd.cut(
        attendance_data["Attendance"],
        bins=[-0.01, 50, 60, 70, 80, 90, 100],
        labels=[
            "0-50",
            "51-60",
            "61-70",
            "71-80",
            "81-90",
            "91-100",
        ],
    )

    attendance_summary = (
        attendance_data.groupby(
            "Attendance Group",
            observed=False,
        )["ML Marks"]
        .agg(["count", "mean", "median"])
        .round(2)
    )

    print(attendance_summary.to_string())

    attendance_summary.to_csv(
        REPORT_DIR / "attendance_group_analysis.csv"
    )

# 7. Check feature correlations using rows with available values
print("\n7. NUMERIC CORRELATION MATRIX")

numeric_data = df.select_dtypes(include=["number"])

correlation_matrix = numeric_data.corr().round(4)

print(correlation_matrix.to_string())

correlation_matrix.to_csv(
    REPORT_DIR / "numeric_correlation_matrix.csv"
)

# 8. Save a concise report
report_file = REPORT_DIR / "data_quality_investigation.txt"

report_lines = [
    "EDUPREDICT AI - DATA QUALITY INVESTIGATION",
    "=" * 50,
    f"Dataset rows: {len(df)}",
    f"Dataset columns: {len(df.columns)}",
    f"Unique ML Marks values: {df['ML Marks'].nunique()}",
    "",
    "ML Marks summary:",
    df["ML Marks"].describe().round(3).to_string(),
    "",
    "Numeric correlation matrix:",
    correlation_matrix.to_string(),
]

report_file.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

print("\n8. REPORTS SAVED")
print(REPORT_DIR / "attendance_group_analysis.csv")
print(REPORT_DIR / "numeric_correlation_matrix.csv")
print(report_file)

print("\nDATA QUALITY INVESTIGATION COMPLETED!")
