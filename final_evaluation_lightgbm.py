# ==========================================
# Final Evaluation of Tuned LightGBM Model
# AI4I Predictive Maintenance
# ==========================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("project1_feature_engineered.csv")

# ==========================================
# Encode Categorical Feature
# ==========================================
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# ==========================================
# Features and Target
# ==========================================
X = df.drop(["Machine failure", "UDI", "Product ID"], axis=1)
y = df["Machine failure"]

# ==========================================
# Clean Column Names
# ==========================================
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

# ==========================================
# Train-Test Split
# ==========================================
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
# Predictions
# ==========================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# Evaluation Metrics
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("=" * 60)
print("FINAL EVALUATION OF TUNED LIGHTGBM MODEL")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Tuned LightGBM")
plt.show()

# ==========================================
# ROC Curve
# ==========================================
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Tuned LightGBM")
plt.show()