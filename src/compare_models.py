from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
print("Loading dataset...")

if not DATA_PATH.exists():
    fallback_path = Path(r"C:\Users\tirun\Downloads\data\ai4i2020.csv")
    if fallback_path.exists():
        DATA_PATH = fallback_path
    else:
        raise FileNotFoundError(
            f"Dataset not found. Update DATA_PATH to your CSV location.\n"
            f"Checked: {DATA_PATH}\n"
            f"Also checked: {fallback_path}"
        )

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Select features and target
feature_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]

target_column = "Machine failure"

X = df[feature_columns].copy()
y = df[target_column].astype(int)

# Encode machine type
X = pd.get_dummies(
    X,
    columns=["Type"],
    drop_first=True,
    dtype=int
)

# Keep the same seven features as previous days
expected_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
]

X = X.reindex(columns=expected_columns, fill_value=0)

# Split the data consistently across all four models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

# Define the models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
         ("model", LogisticRegression(
           solver="liblinear",
           class_weight="balanced",
           max_iter=2000,
           random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
     "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
}

# Train and evaluate every model
results = []

for name, model in models.items():
    print("\n" + "=" * 55)
    print("Training:", name)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Alarms": fp,
        "Missed Failures": fn,
        "Detected Failures": tp
    })

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"False alarms:      {fp}")
    print(f"Missed failures:   {fn}")
    print(f"Detected failures: {tp}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0
    ))

# Compare results and save the report
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-score",
    ascending=False
)

output_path = REPORTS_DIR / "day6_model_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print("DAY 6 MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print(f"\nResults saved to: {output_path}")
print("\nAll models above use their default classification threshold.")