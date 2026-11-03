from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================================================
# EDUPREDICT AI - DATA PREPROCESSING
# =========================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "raw" / "student_performance.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("EDUPREDICT AI - DATA PREPROCESSING")
print("=" * 65)

# 1. Load the dataset
df = pd.read_csv(RAW_FILE)

print("\n1. Original dataset shape:", df.shape)

# 2. Remove duplicate rows
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicates_before)

# 3. Convert date column to datetime
df["Join Date"] = pd.to_datetime(
    df["Join Date"],
    errors="coerce"
)

# 4. Remove rows where the target is missing
target = "ML Marks"

missing_target = df[target].isna().sum()
df = df.dropna(subset=[target]).copy()

print("Rows removed because ML Marks is missing:", missing_target)

# 5. Define model features
# Exclude personal identifiers and the date.
excluded_columns = [
    "Student ID",
    "Student Name",
    "Join Date",
    "ML Marks",
]

X = df.drop(columns=excluded_columns)
y = df[target].astype(float)

# 6. Identify numeric and categorical features
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\n2. Numeric features:")
print(numeric_features)

print("\n3. Categorical features:")
print(categorical_features)

# 7. Create preprocessing pipelines
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

# 8. Split data before fitting preprocessing.
# This prevents preprocessing statistics from leaking
# from the test set into the training process.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\n4. Training records:", len(X_train))
print("Testing records:", len(X_test))

# 9. Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 10. Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
)

test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
)

# Add target column to each processed dataset
train_processed[target] = y_train.reset_index(drop=True)
test_processed[target] = y_test.reset_index(drop=True)

# 11. Save datasets
train_file = PROCESSED_DIR / "train_processed.csv"
test_file = PROCESSED_DIR / "test_processed.csv"

train_processed.to_csv(train_file, index=False)
test_processed.to_csv(test_file, index=False)

# Save the original feature splits as well.
# These preserve categorical values for later model pipelines.
X_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False)
y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

# Save a text report
report = [
    "EDUPREDICT AI - PREPROCESSING REPORT",
    "=" * 45,
    f"Original records: {len(pd.read_csv(RAW_FILE))}",
    f"Records after cleaning: {len(df)}",
    f"Target: {target}",
    f"Training records: {len(X_train)}",
    f"Testing records: {len(X_test)}",
    f"Numeric features: {numeric_features}",
    f"Categorical features: {categorical_features}",
    f"Processed training shape: {X_train_processed.shape}",
    f"Processed testing shape: {X_test_processed.shape}",
    f"Missing values in processed training data: {train_processed.isnull().sum().sum()}",
    f"Missing values in processed testing data: {test_processed.isnull().sum().sum()}",
]

report_text = "\n".join(report)

(REPORT_DIR / "preprocessing_report.txt").write_text(
    report_text,
    encoding="utf-8",
)

print("\n5. PREPROCESSING RESULTS")
print(report_text)

print("\n6. FILES CREATED")
print(train_file)
print(test_file)
print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
