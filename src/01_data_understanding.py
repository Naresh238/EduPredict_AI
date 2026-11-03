from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "student_performance.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

print("=" * 65)
print("EDUPREDICT AI - DATA UNDERSTANDING AND EDA")
print("=" * 65)

print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n2. DATA TYPES")
print(df.dtypes.to_string())

print("\n3. MISSING VALUES")
missing = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().mean() * 100).round(2)
})
print(missing.to_string())
missing.to_csv(REPORT_DIR / "missing_values_report.csv")

print("\n4. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n5. NUMERIC SUMMARY")
print(df.describe().round(2).to_string())
df.describe().round(2).to_csv(REPORT_DIR / "numeric_summary.csv")

print("\n6. CATEGORICAL SUMMARY")
for column in df.select_dtypes(include=["object"]).columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(10).to_string())

print("\n7. VALIDATION CHECKS")
for column in ["Attendance", "Python Marks", "ML Marks"]:
    if column in df.columns:
        invalid = df[column].dropna().between(0, 100).eq(False).sum()
        print(f"{column}: {invalid} non-missing values outside 0-100")

if "Year" in df.columns:
    print("Academic year values:", sorted(df["Year"].dropna().unique()))

# Plot 1: Missing values
plt.figure(figsize=(10, 5))
missing["Missing_Count"].sort_values(ascending=False).plot(kind="bar")
plt.title("Missing Values by Column")
plt.xlabel("Column")
plt.ylabel("Missing count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "missing_values.png", dpi=150)
plt.close()

# Plot 2: Numeric distributions
numeric_columns = [
    c for c in ["Age", "Attendance", "Python Marks",
                "ML Marks", "Projects Completed"]
    if c in df.columns
]

for column in numeric_columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{column.lower().replace(' ', '_')}_distribution.png",
                dpi=150)
    plt.close()

# Plot 3: Attendance vs ML marks
if "Attendance" in df.columns and "ML Marks" in df.columns:
    plot_df = df[["Attendance", "ML Marks"]].dropna()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=plot_df, x="Attendance", y="ML Marks", alpha=0.6)
    plt.title("Attendance vs Machine Learning Marks")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "attendance_vs_ml_marks.png", dpi=150)
    plt.close()

# Plot 4: Average marks by department
if "Department" in df.columns and "ML Marks" in df.columns:
    department_means = (
        df.groupby("Department", dropna=False)["ML Marks"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n8. AVERAGE ML MARKS BY DEPARTMENT")
    print(department_means.round(2).to_string())

    department_means.to_csv(REPORT_DIR / "department_average_ml_marks.csv")

    plt.figure(figsize=(8, 5))
    department_means.plot(kind="bar")
    plt.title("Average ML Marks by Department")
    plt.xlabel("Department")
    plt.ylabel("Average ML marks")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "average_ml_marks_by_department.png", dpi=150)
    plt.close()

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Reports saved to:", REPORT_DIR)
print("Charts saved to:", FIGURE_DIR)
