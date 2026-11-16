
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Load the trained Random Forest model
model = joblib.load("models/random_forest.joblib")

# Get predicted probabilities of machine failure
failure_probabilities = model.predict_proba(X_test)[:, 1]

# Test different thresholds
thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

results = []

for threshold in thresholds:

    predictions = (
        failure_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Alarms": fp,
        "Missed Failures": fn,
        "Detected Failures": tp
    })

results_df = pd.DataFrame(results)

print("\nRANDOM FOREST THRESHOLD COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "reports/day4_threshold_comparison.csv",
    index=False
)

print("\nResults saved to reports/day4_threshold_comparison.csv")