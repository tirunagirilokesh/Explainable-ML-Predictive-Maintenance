
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# 1. Define project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the preprocessed data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
print("Training failure counts:")
print(y_train.value_counts())


# 3. Define two models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
   ),
}


# 4. Train and evaluate each model
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)
    trained_models[name] = model

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("\nMetrics:")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")


# 5. Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORT_DIR / "day4_model_comparison.csv",
    index=False,
)

# 6. Save all four trained models
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    trained_models["Logistic Regression"],
    MODEL_DIR / "logistic_regression.joblib",
)

joblib.dump(
    trained_models["Random Forest"],
    MODEL_DIR / "random_forest.joblib",
)

joblib.dump(
    trained_models["Decision Tree"],
    MODEL_DIR / "decision_tree.joblib",
)

joblib.dump(
    trained_models["Gradient Boosting"],
    MODEL_DIR / "gradient_boosting.joblib",
)

print("\nAll four trained models saved successfully.")
print(f"Model directory: {MODEL_DIR}")

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))
print("\n=== DAY 4 COMPLETE ===")
print("Comparison saved to:", REPORT_DIR / "day4_model_comparison.csv")
print("Models saved in:", MODEL_DIR)
