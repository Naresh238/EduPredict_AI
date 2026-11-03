from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external" / "uci_student_performance" / "extracted"

for filename in ["student-mat.csv", "student-por.csv"]:
    filepath = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"DATASET: {filename}")
    print("=" * 65)

    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        continue

    df = pd.read_csv(filepath, sep=";")

    print(f"Dataset shape: {df.shape}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nFinal grade (G3) summary:")
    print(df["G3"].describe().round(2).to_string())

print("\nDATASET INSPECTION COMPLETED!")
