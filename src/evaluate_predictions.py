# ==========================================
# DAY 8: PREDICTION EVALUATION
# ==========================================

from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. FILE PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = ROOT / "reports" / "day8_prediction_report.csv"
REPORT_DIR = ROOT / "reports"

print("\nLoading Day 8 prediction report...")

df = pd.read_csv(REPORT_PATH)

# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "Actual_Failure",
    "Predicted_Failure",
    "Failure_Probability"
]

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in prediction report: {missing}"
    )

y_true = df["Actual_Failure"].astype(int)
y_pred = df["Predicted_Failure"].astype(int)
y_prob = df["Failure_Probability"]

# ==========================================
# 3. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true, y_pred, zero_division=0
)

recall = recall_score(
    y_true, y_pred, zero_division=0
)

f1 = f1_score(
    y_true, y_pred, zero_division=0
)

# ==========================================
# 4. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n====================================")
print("DAY 8 MODEL EVALUATION")
print("====================================")

print(f"Total records       : {len(df)}")
print(f"Accuracy             : {accuracy:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1-score             : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTrue Negatives       :", tn)
print("False Positives      :", fp)
print("False Negatives      :", fn)
print("True Positives       :", tp)

# ==========================================
# 5. ROC-AUC AND PR-AUC
# ==========================================

if y_true.nunique() == 2:

    roc_auc = roc_auc_score(y_true, y_prob)

    pr_auc = average_precision_score(
        y_true, y_prob
    )

    print(f"\nROC-AUC              : {roc_auc:.4f}")
    print(f"Average Precision    : {pr_auc:.4f}")

else:
    roc_auc = None
    pr_auc = None

    print("\nROC-AUC and PR-AUC require both classes.")

# ==========================================
# 6. SAVE CONFUSION MATRIX
# ==========================================

cm_df = pd.DataFrame(
    cm,
    index=["Actual_Normal", "Actual_Failure"],
    columns=["Predicted_Normal", "Predicted_Failure"]
)

cm_df.to_csv(
    REPORT_DIR / "day8_confusion_matrix.csv"
)

# ==========================================
# 7. SAVE CLASSIFICATION REPORT
# ==========================================

class_report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Failure"],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(class_report).transpose().to_csv(
    REPORT_DIR / "day8_classification_report.csv"
)

# ==========================================
# 8. SAVE SUMMARY METRICS
# ==========================================

metrics = {
    "Total Records": len(df),
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "True Negatives": tn,
    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc
}

pd.DataFrame(
    [metrics]
).to_csv(
    REPORT_DIR / "day8_evaluation_metrics.csv",
    index=False
)

print("\nReports saved successfully!")

print("1. reports/day8_confusion_matrix.csv")
print("2. reports/day8_classification_report.csv")
print("3. reports/day8_evaluation_metrics.csv")

print("\nDAY 8 EVALUATION COMPLETED")