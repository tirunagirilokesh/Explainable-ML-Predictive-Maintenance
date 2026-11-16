
from pathlib import Path
import joblib

# Project model directory
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

for model_name, filename in model_files.items():
    model_path = MODELS_DIR / filename

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    if not model_path.exists():
        print(f"File not found: {model_path}")
        continue

    try:
        model = joblib.load(model_path)

        print("\n1. Model type:")
        print(type(model))

        print("\n2. Model configuration:")
        print(model)

        print("\n3. Expected input features:")
        if hasattr(model, "feature_names_in_"):
            print(model.feature_names_in_)
        else:
            print("Feature names are not stored in this model.")

        print("\n4. Number of input features:")
        if hasattr(model, "n_features_in_"):
            print(model.n_features_in_)

        print("\n5. Classes:")
        if hasattr(model, "classes_"):
            print(model.classes_)

        print("\n6. Model parameters:")
        print(model.get_params())

        print("\n7. Learned coefficients or feature importance:")

        if hasattr(model, "coef_"):
            print("Coefficients:")
            print(model.coef_)

        if hasattr(model, "feature_importances_"):
            print("Feature importances:")
            print(model.feature_importances_)

        print("\n8. Model inspection completed.")

    except Exception as exc:
        print(f"Could not inspect model: {exc}")

print("\nAll available models have been inspected.")

