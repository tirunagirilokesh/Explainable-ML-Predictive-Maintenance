# ==========================================
# DAY 5: EXPLAINABLE AI
# Explainable Machine Learning for Predictive Maintenance
# ==========================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ==========================================
# 1. DEFINE FILE PATHS
# ==========================================

MODEL_PATH = "models/random_forest.joblib"

X_TEST_PATH = "data/processed/X_test.csv"

REPORT_DIR = "reports"

os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD TEST DATA
# ==========================================

print("\nLoading test dataset...")

X_test = pd.read_csv(X_TEST_PATH)

print("Test dataset shape:", X_test.shape)

print("\nFeatures:")
print(X_test.columns.tolist())

# ==========================================
# 3. LOAD TRAINED RANDOM FOREST MODEL
# ==========================================

print("\nLoading trained Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# ==========================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("\nCalculating feature importance...")

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n=== FEATURE IMPORTANCE ===")
print(importance_df.to_string(index=False))

# Save feature importance results

importance_df.to_csv(
    os.path.join(REPORT_DIR, "day5_feature_importance.csv"),
    index=False
)

# Plot feature importance

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"][::-1],
    importance_df["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")

plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "day5_feature_importance.png"),
    dpi=300
)

plt.show()

# ==========================================
# 5. SHAP EXPLANATIONS
# ==========================================

print("\nCalculating SHAP values...")

# TreeExplainer is designed for tree-based models.

explainer = shap.TreeExplainer(model)

# Explain the test dataset.

shap_values = explainer.shap_values(X_test)

# SHAP versions can return different shapes for
# binary classification. Select the failure class.

if isinstance(shap_values, list):
    # Older SHAP versions:
    # Each list item represents one output class.
    if len(shap_values) == 2:
        failure_shap_values = shap_values[1]
    else:
        failure_shap_values = shap_values[0]

elif getattr(shap_values, "ndim", 0) == 3:
    # Some versions return:
    # samples x features x classes
    if shap_values.shape[-1] == 2:
        failure_shap_values = shap_values[:, :, 1]
    else:
        failure_shap_values = shap_values[:, :, 0]

else:
    # Binary Random Forest models commonly return
    # a 2D SHAP array in newer versions.
    failure_shap_values = shap_values

print("SHAP values calculated successfully!")

print("SHAP values shape:", failure_shap_values.shape)

# ==========================================
# 6. SHAP SUMMARY PLOT
# ==========================================

print("\nGenerating SHAP summary plot...")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    failure_shap_values,
    X_test,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "day5_shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 7. SHAP BAR PLOT
# ==========================================

print("\nGenerating SHAP importance bar plot...")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    failure_shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "day5_shap_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# 8. COMPLETE
# ==========================================

print("\n====================================")
print("DAY 5 EXPLAINABILITY COMPLETED")
print("====================================")

print("\nGenerated files:")

print("1. reports/day5_feature_importance.csv")
print("2. reports/day5_feature_importance.png")
print("3. reports/day5_shap_summary.png")
print("4. reports/day5_shap_importance.png")