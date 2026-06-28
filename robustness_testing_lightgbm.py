# =====================================================
# Robustness Testing using 5-Fold Cross Validation
# Tuned LightGBM Model
# =====================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("project1_feature_engineered.csv")

# Encode categorical column
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# Features and Target
X = df.drop(["Machine failure", "UDI", "Product ID"], axis=1)
y = df["Machine failure"]

# Clean feature names
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

# Load tuned model
model = joblib.load("lightgbm_tuned_model.pkl")

# 5-Fold Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracy_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

f1_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("=" * 60)
print("ROBUSTNESS TESTING USING 5-FOLD CROSS VALIDATION")
print("=" * 60)

print("\nAccuracy Scores")
print(accuracy_scores)

print("\nMean Accuracy :", accuracy_scores.mean())
print("Std Accuracy  :", accuracy_scores.std())

print("\nF1 Scores")
print(f1_scores)

print("\nMean F1 Score :", f1_scores.mean())
print("Std F1 Score  :", f1_scores.std())