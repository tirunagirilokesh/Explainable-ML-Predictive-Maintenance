
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic configuration
sns.set_theme(style="whitegrid")

print("=" * 50)
print("DAY 2: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Dataset overview
print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# 2. Machine failure distribution
failure_counts = df["Machine failure"].value_counts().sort_index()

print("\nMachine failure counts:")
print(failure_counts)

print("\nMachine failure percentages:")
print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

# 3. Visualize failure distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Observations")
plt.tight_layout()
plt.savefig(REPORT_DIR / "failure_distribution.png", dpi=300)
plt.show()

# 4. Failure rate by machine type
failure_rate = df.groupby("Type")["Machine failure"].mean() * 100

print("\nFailure rate by machine type (%):")
print(failure_rate.round(2))

plt.figure(figsize=(7, 5))
sns.barplot(x=failure_rate.index, y=failure_rate.values)
plt.title("Machine Failure Rate by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Failure Rate (%)")
plt.tight_layout()
plt.savefig(REPORT_DIR / "failure_by_type.png", dpi=300)
plt.show()

# 5. Sensor readings compared with machine failure
sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for column in sensor_columns:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x="Machine failure",
        y=column
    )

    plt.title(f"{column} vs Machine Failure")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel(column)
    plt.tight_layout()

    filename = column.lower()
    filename = filename.replace(" ", "_")
    filename = filename.replace("[", "").replace("]", "")

    plt.savefig(
        REPORT_DIR / f"{filename}_by_failure.png",
        dpi=300
    )
    plt.show()

# 6. Correlation between numerical features
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 9))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=300)
plt.show()

# 7. Save a text report
report_path = REPORT_DIR / "day2_findings.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("DAY 2: EXPLORATORY DATA ANALYSIS\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Dataset rows: {df.shape[0]}\n")
    file.write(f"Dataset columns: {df.shape[1]}\n")
    file.write(f"Missing values: {df.isnull().sum().sum()}\n")
    file.write(f"Duplicate rows: {df.duplicated().sum()}\n\n")

    file.write("Machine failure counts:\n")
    file.write(f"{failure_counts.to_string()}\n\n")

    file.write("Failure percentages:\n")
    file.write(
        f"{(df['Machine failure'].value_counts(normalize=True) * 100).round(2).to_string()}\n\n"
    )

    file.write("Failure rate by machine type (%):\n")
    file.write(f"{failure_rate.round(2).to_string()}\n\n")

    file.write("Numerical feature correlations with Machine failure:\n")
    file.write(
        f"{numeric_df.corr()['Machine failure'].sort_values(ascending=False).to_string()}\n"
    )

print("\nEDA completed successfully!")
print("Graphs and report saved in the reports folder.")
print("Report:", report_path)
