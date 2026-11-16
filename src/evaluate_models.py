
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Project paths
MODEL_DIR = "models"
DATA_DIR = "data/processed"
REPORT_DIR = "reports"

# Load test data
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

# Load saved models
logistic_model = joblib.load(
    f"{MODEL_DIR}/logistic_regression.joblib"
)

random_forest_model = joblib.load(
    f"{MODEL_DIR}/random_forest.joblib"
)

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

results = []

for model_name, model in models.items():

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    # Predict machine failure
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
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

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

# Save evaluation results
comparison = pd.DataFrame(results)

comparison.to_csv(
    f"{REPORT_DIR}/day4_local_evaluation.csv",
    index=False
)

print("\nEvaluation completed!")
print("Results saved to reports/day4_local_evaluation.csv")
