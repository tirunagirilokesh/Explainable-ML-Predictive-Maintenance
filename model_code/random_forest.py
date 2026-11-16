
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

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
    # Load existing datasets
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

    # Maintain the feature order expected by the model
    X_train = X_train[FEATURES]
    X_test = X_test[FEATURES]

    # Recreate the saved model configuration
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate on the existing test set
    predictions = model.predict(X_test)

    print("\nRANDOM FOREST EVALUATION")
    print("=" * 45)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save feature importance as a report
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    # Save outputs without overwriting existing files
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    output_path = MODEL_DIR / "random_forest_retrained.joblib"
    joblib.dump(model, output_path)

    report_path = (
        PROJECT_DIR / "reports" / "random_forest_retrained_importance.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(report_path, index=False)

    print(f"\nNew model saved to: {output_path}")
    print(f"Feature importance saved to: {report_path}")


if __name__ == "__main__":
    main()