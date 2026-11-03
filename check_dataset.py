from pathlib import Path
import pandas as pd

file_path = Path(__file__).parent / "data" / "raw" / "student_performance.csv"

print("=" * 55)
print("EDUPREDICT AI - DATASET VERIFICATION")
print("=" * 55)

if not file_path.exists():
    print("ERROR: Dataset not found:", file_path)
else:
    df = pd.read_csv(file_path)

    print(f"\nDataset file: {file_path}")
    print(f"Total student records: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:", df.duplicated().sum())

    if len(df) > 2000:
        print("\nSUCCESS: More than 2,000 records verified!")
    else:
        print("\nWARNING: This dataset contains 2,000 or fewer records.")
