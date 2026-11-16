
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
]


def main():
    # Load datasets
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    # Remove accidental index columns
    X_train = X_train.loc[
        :, ~X_train.columns.astype(str).str.startswith("Unnamed")
    ]
    X_test = X_test.loc[
        :, ~X_test.columns.astype(str).str.startswith("Unnamed")
    ]

    # Maintain consistent feature order
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]

    # Build Logistic Regression pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluation
    print("\nLOGISTIC REGRESSION EVALUATION")
    print("=" * 50)
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )
    print(
        f"Failure Precision: "
        f"{precision_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Failure Recall:    "
        f"{recall_score(y_test, predictions, zero_division=0):.4f}"
    )
    print(
        f"Failure F1-score:  "
        f"{f1_score(y_test, predictions, zero_division=0):.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "logistic_regression_retrained.joblib"
    joblib.dump(model, model_path)

    # Save evaluation report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame([{
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Failure Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Failure Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Failure F1": f1_score(
            y_test, predictions, zero_division=0
        ),
    }])

    report_path = REPORT_DIR / "logistic_regression_retrained_metrics.csv"
    report.to_csv(report_path, index=False)

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {report_path}")


if __name__ == "__main__":
    main()