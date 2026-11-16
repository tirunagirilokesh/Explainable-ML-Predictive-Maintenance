
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
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

MODELS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting.joblib",
}

def main():
    # Load test data
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

    # Remove accidental index columns
    X_test = X_test.loc[
        :, ~X_test.columns.astype(str).str.startswith("Unnamed")
    ]

    X_test = X_test[FEATURES]

    results = []

    print("\nFOUR-MODEL COMPARISON")
    print("=" * 65)

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\nSkipping {model_name}: model file not found.")
            print(f"Expected location: {model_path}")
            continue

        model = joblib.load(model_path)

        predictions = model.predict(X_test)

        # Probability estimates are used for ROC-AUC when available
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)
        mcc = matthews_corrcoef(y_test, predictions)

        roc_auc = (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None
            else float("nan")
        )

        # Confusion matrix: rows are actual, columns are predicted
        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        specificity = (
            tn / (tn + fp) if (tn + fp) > 0 else 0.0
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Balanced Accuracy": balanced_accuracy,
            "Failure Precision": precision,
            "Failure Recall": recall,
            "Failure F1": f1,
            "Specificity": specificity,
            "MCC": mcc,
            "ROC-AUC": roc_auc,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "True Positives": tp,
        })

        print(f"\n{model_name}")
        print("-" * 45)
        print(f"Accuracy:          {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Failure Precision: {precision:.4f}")
        print(f"Failure Recall:    {recall:.4f}")
        print(f"Failure F1-score:  {f1:.4f}")
        print(f"Specificity:       {specificity:.4f}")
        print(f"MCC:               {mcc:.4f}")
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    if not results:
        print("\nNo models were evaluated. Check the model paths.")
        return

    # Rank by failure F1-score for a balanced precision/recall view.
    # Also inspect failure recall because missed failures can be costly.
    comparison = pd.DataFrame(results).sort_values(
        by="Failure F1",
        ascending=False,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "four_model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\n\nFINAL COMPARISON TABLE")
    print("=" * 100)
    print(comparison.round(4).to_string(index=False))

    print(f"\nComparison report saved to:\n{output_path}")
    print("\nInterpret the results using failure recall, precision, F1,")
    print("MCC, ROC-AUC, and the confusion matrix—not accuracy alone.")


if __name__ == "__main__":
    main()