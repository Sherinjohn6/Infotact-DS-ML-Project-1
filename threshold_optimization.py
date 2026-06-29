import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

# Load SMOTE dataset
df = pd.read_csv("smote_dataset.csv")

# Clean column names for LightGBM
df.columns = [
    re.sub(r'[^A-Za-z0-9_]', '_', col).strip('_')
    for col in df.columns
]
df.columns = [re.sub(r'_+', '_', col) for col in df.columns]

# Features and target
X = df.drop("Machine_failure", axis=1)
y = df["Machine_failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train LightGBM
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Predict probabilities
y_prob = lgb_model.predict_proba(X_test)[:, 1]

# Threshold Optimization
thresholds = np.arange(0.10, 1.00, 0.01)

best_threshold = 0.50
best_f1 = 0
f1_scores = []

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    score = f1_score(y_test, y_pred)
    f1_scores.append(score)

    if score > best_f1:
        best_f1 = score
        best_threshold = t

print("=" * 50)
print(f"Best Threshold : {best_threshold:.2f}")
print(f"Best F1 Score  : {best_f1:.4f}")
print("=" * 50)

# Final prediction
y_pred_best = (y_prob >= best_threshold).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_best))

print("\nClassification Report")
print(classification_report(y_test, y_pred_best))

# Plot
plt.figure(figsize=(8,5))
plt.plot(thresholds, f1_scores, marker='o')
plt.axvline(best_threshold, color='red', linestyle='--',
            label=f'Best Threshold = {best_threshold:.2f}')
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold Optimization - LightGBM")
plt.legend()
plt.grid(True)
plt.show()