# ==========================================
# DAY 8: MACHINE FAILURE PREDICTION REPORT
# Explainable Machine Learning for Predictive Maintenance
# ==========================================

from pathlib import Path
import joblib
import pandas as pd

# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "random_forest.joblib"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORT_DIR / "day8_prediction_report.csv"

# ==========================================
# 2. LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

X_test = pd.read_csv(X_TEST_PATH)

print("Test dataset shape:", X_test.shape)

# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

print("\nLoading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# 4. GENERATE PREDICTIONS
# ==========================================

print("\nPredicting machine failures...")

predictions = model.predict(X_test)

# Probability assigned to the failure class
classes = list(model.classes_)

if 1 not in classes:
    raise ValueError(
        "The trained model does not contain failure class 1."
    )

failure_index = classes.index(1)

probabilities = model.predict_proba(X_test)[:, failure_index]

# ==========================================
# 5. CREATE PREDICTION REPORT
# ==========================================

report = X_test.copy()

report.insert(
    0,
    "Machine_ID",
    [f"Machine_{i}" for i in range(1, len(X_test) + 1)]
)

report["Predicted_Failure"] = predictions
report["Failure_Probability"] = probabilities
report["Failure_Probability_Percent"] = (
    probabilities * 100
).round(2)

report["Predicted_Status"] = report[
    "Predicted_Failure"
].map({
    0: "Normal",
    1: "Failure Predicted"
})

# ==========================================
# 6. ADD ACTUAL LABELS IF AVAILABLE
# ==========================================

if Y_TEST_PATH.exists():

    print("\nLoading actual test labels...")

    y_test = pd.read_csv(Y_TEST_PATH)

    # Support a CSV containing one target column
    if "Machine failure" in y_test.columns:
        actual = y_test["Machine failure"]
    elif y_test.shape[1] == 1:
        actual = y_test.iloc[:, 0]
    else:
        raise ValueError(
            "Cannot identify the actual target column in y_test.csv."
        )

    if len(actual) != len(report):
        raise ValueError(
            "Test features and labels have different row counts."
        )

    report["Actual_Failure"] = actual.to_numpy()

    report["Prediction_Result"] = (
        report["Predicted_Failure"]
        == report["Actual_Failure"]
    ).map({
        True: "Correct",
        False: "Incorrect"
    })

    print("Actual labels added to the report.")

else:
    print(
        "\ny_test.csv not found. "
        "Generating predictions without actual labels."
    )

# ==========================================
# 7. SAVE REPORT
# ==========================================

report.to_csv(OUTPUT_PATH, index=False)

print("\n====================================")
print("DAY 8 PREDICTION REPORT COMPLETED")
print("====================================")

print("\nFirst five predictions:")
print(report.head().to_string(index=False))

print("\nPredicted failure count:",
      int((predictions == 1).sum()))

print("Predicted normal count:",
      int((predictions == 0).sum()))

print("\nReport saved to:")
print(OUTPUT_PATH)