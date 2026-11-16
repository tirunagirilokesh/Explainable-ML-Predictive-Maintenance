from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"

OUTPUT_PATH = REPORTS_DIR / "day7_threshold_comparison.csv"

print("=" * 65)
print("DAY 7: THRESHOLD COMPARISON")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD TRAINING DATA
# --------------------------------------------------

print("\nLoading training data...")

X = pd.read_csv(X_TRAIN_PATH)
y = pd.read_csv(Y_TRAIN_PATH).squeeze().astype(int)

print("Training features:", X.shape)
print("Training targets:", y.shape)

# --------------------------------------------------
# 3. CREATE TRAINING AND VALIDATION SETS
# --------------------------------------------------
# We use part of the training data for validation.
# The original test dataset remains untouched.

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nInternal training features:", X_train.shape)
print("Validation features:", X_val.shape)

# --------------------------------------------------
# 4. DEFINE THE MODELS
# --------------------------------------------------

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    ),
}

# --------------------------------------------------
# 5. TEST DIFFERENT CLASSIFICATION THRESHOLDS
# --------------------------------------------------

thresholds = [0.50, 0.40, 0.30, 0.20, 0.10]

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    # Probability that each machine will fail
    failure_probabilities = model.predict_proba(X_val)[:, 1]

    for threshold in thresholds:

        # Convert probabilities into 0 or 1 predictions
        y_pred = (
            failure_probabilities >= threshold
        ).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_val, y_pred)

        precision = precision_score(
            y_val, y_pred, zero_division=0
        )

        recall = recall_score(
            y_val, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_val, y_pred, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            y_pred,
            labels=[0, 1],
        ).ravel()

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-score": round(f1, 4),
            "False Alarms": int(fp),
            "Missed Failures": int(fn),
            "Detected Failures": int(tp),
        })

# --------------------------------------------------
# 6. DISPLAY AND SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Model", "Threshold"],
    ascending=[True, False],
)

print("\n" + "=" * 65)
print("DAY 7 THRESHOLD COMPARISON RESULTS")
print("=" * 65)

print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)

print("\nResults saved successfully!")
print(f"Report location: {OUTPUT_PATH}")

print("\nRemember:")
print("- Lower thresholds usually detect more failures.")
print("- Lower thresholds can also produce more false alarms.")
print("- Compare missed failures, precision, recall, and F1-score.")
print("- The original test dataset has not been used in this analysis.")