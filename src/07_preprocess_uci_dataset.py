from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
    ROOT
    / "data"
    / "external"
    / "uci_student_performance"
    / "extracted"
    / "student-mat.csv"
)

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("EDUPREDICT AI - UCI DATA PREPROCESSING")
print("=" * 70)

# 1. Load the official UCI Mathematics dataset.
if not SOURCE_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE_FILE}"
    )

df = pd.read_csv(SOURCE_FILE, sep=";")

print("\n1. ORIGINAL DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Check data quality.
print("\n2. DATA QUALITY")

print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

# The UCI Mathematics dataset contains no missing values.
# Remove duplicate records only if they exist.
df = df.drop_duplicates().copy()

# 3. Define the prediction target.
target = "G3"

# Exclude G1 and G2 to evaluate prediction without earlier grades.
excluded_columns = ["G1", "G2", target]

X = df.drop(columns=excluded_columns)
y = df[target]

print("\n3. PREDICTION SETUP")
print("Target:", target)
print("Target meaning: Final mathematics grade (0-20)")
print("Excluded columns:", ["G1", "G2"])
print("Feature count:", X.shape[1])
print("Target count:", len(y))

# 4. Create a separate student-risk label.
# This label is for analysis and classification later.
df["Risk_Status"] = (
    df["G3"]
    .lt(10)
    .map({
        True: "At Risk",
        False: "Not At Risk",
    })
)

print("\n4. STUDENT RISK DISTRIBUTION")
print(df["Risk_Status"].value_counts().to_string())

risk_summary = (
    df["Risk_Status"]
    .value_counts()
    .rename_axis("Risk_Status")
    .reset_index(name="Student_Count")
)

risk_summary["Percentage"] = (
    risk_summary["Student_Count"] / len(df) * 100
).round(2)

risk_summary.to_csv(
    REPORT_DIR / "student_risk_distribution.csv",
    index=False,
)

# 5. Split data into training and testing sets.
# Stratification is used for the derived risk categories so that
# both sets have a similar proportion of at-risk students.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=df["Risk_Status"],
)

print("\n5. TRAIN-TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# 6. Save reproducible datasets.
X_train.to_csv(
    PROCESSED_DIR / "uci_X_train.csv",
    index=False,
)

X_test.to_csv(
    PROCESSED_DIR / "uci_X_test.csv",
    index=False,
)

y_train.to_frame(name=target).to_csv(
    PROCESSED_DIR / "uci_y_train.csv",
    index=False,
)

y_test.to_frame(name=target).to_csv(
    PROCESSED_DIR / "uci_y_test.csv",
    index=False,
)

# Save the cleaned dataset with the derived risk label.
df.to_csv(
    PROCESSED_DIR / "uci_student_math_clean.csv",
    index=False,
)

# Save a concise preprocessing report.
report = [
    "EDUPREDICT AI - UCI PREPROCESSING REPORT",
    "=" * 50,
    f"Source: {SOURCE_FILE.name}",
    f"Clean dataset rows: {len(df)}",
    f"Feature count: {X.shape[1]}",
    f"Target: {target}",
    "Target range: 0-20",
    "Excluded predictors: G1, G2",
    "Risk definition: G3 < 10",
    f"Training rows: {len(X_train)}",
    f"Testing rows: {len(X_test)}",
    f"Duplicate rows after cleaning: {df.duplicated().sum()}",
    f"Missing values after cleaning: {df.isnull().sum().sum()}",
]

report_file = REPORT_DIR / "uci_preprocessing_report.txt"
report_file.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("\n6. FILES SAVED")
print(PROCESSED_DIR / "uci_X_train.csv")
print(PROCESSED_DIR / "uci_X_test.csv")
print(PROCESSED_DIR / "uci_y_train.csv")
print(PROCESSED_DIR / "uci_y_test.csv")
print(PROCESSED_DIR / "uci_student_math_clean.csv")
print(REPORT_DIR / "student_risk_distribution.csv")
print(report_file)

print("\nUCI PREPROCESSING COMPLETED SUCCESSFULLY!")
