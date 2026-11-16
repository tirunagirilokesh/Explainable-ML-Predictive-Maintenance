
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "ai4i2020.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the dataset
df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 3. Remove identifier columns
df = df.drop(columns=["UDI", "Product ID"])

# 4. Define the target variable
y = df["Machine failure"]

# 5. Select useful input features
X = df.drop(
    columns=[
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]
)

# 6. Convert categorical Type into numerical columns
X = pd.get_dummies(
    X,
    columns=["Type"],
    drop_first=True,
    dtype=int
)

print("\nFeatures after preprocessing:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

# 7. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 8. Save the prepared datasets
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

# 9. Display the results
print("\n=== PREPROCESSING COMPLETE ===")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Testing targets:", y_test.shape)

print("\nFiles saved in:", OUTPUT_DIR)
