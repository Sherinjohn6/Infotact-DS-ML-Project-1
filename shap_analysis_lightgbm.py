
# ==========================================
# SHAP Analysis for Tuned LightGBM Model
# AI4I Predictive Maintenance
# ==========================================

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("project1_feature_engineered.csv")

# Encode categorical column
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Features and Target
X = df.drop(["Machine failure", "UDI", "Product ID"], axis=1)
y = df["Machine failure"]

# Clean column names
X.columns = (
    X.columns
    .str.replace(" ", "_", regex=False)
    .str.replace("[", "", regex=False)
    .str.replace("]", "", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "_", regex=False)
    .str.replace("-", "_", regex=False)
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Load Tuned LightGBM Model
# ==========================================
model = joblib.load("lightgbm_tuned_model.pkl")

# ==========================================
# SHAP Explainer
# ==========================================
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# ==========================================
# SHAP Summary Plot
# ==========================================
print("Generating SHAP Summary Plot...")
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=True
)

# ==========================================
# SHAP Beeswarm Plot
# ==========================================
print("Generating SHAP Beeswarm Plot...")
shap.summary_plot(
    shap_values,
    X_test,
    show=True
)

# ==========================================
# Save Feature Importance Plot
# ==========================================
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False
)

plt.savefig("lightgbm_shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSHAP Analysis Completed Successfully!")
print("Feature importance plot saved as:")
print("lightgbm_shap_feature_importance.png")