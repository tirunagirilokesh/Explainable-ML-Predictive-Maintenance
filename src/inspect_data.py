
from pathlib import Path
import pandas as pd

project_dir = Path(r"C:\Users\tirun\Downloads\predictive-maintenance-xai\data\ai4i2020.csv").resolve().parent.parent
dataset_path = project_dir / "data" / "ai4i2020.csv"
report_path = project_dir / "reports" / "day1_report.txt"

# Load the CSV
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Cannot find dataset: {dataset_path}"
    )

df = pd.read_csv(dataset_path)

# Display basic information
print("\n=== DATASET SIZE ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== FIRST FIVE ROWS ===")
print(df.head())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== MACHINE FAILURE COUNTS ===")
print(df["Machine failure"].value_counts().sort_index())

# Save a report
report = (
    "DAY 1: DATASET INSPECTION\n\n"
    f"Rows: {df.shape[0]}\n"
    f"Columns: {df.shape[1]}\n\n"
    f"Column names:\n{df.columns.tolist()}\n\n"
    f"Missing values:\n{df.isnull().sum().to_string()}\n\n"
    f"Machine failure counts:\n"
    f"{df['Machine failure'].value_counts().sort_index().to_string()}\n"
)

report_path.write_text(report, encoding="utf-8")

print(f"\nReport saved to: {report_path}")
