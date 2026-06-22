import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier

# Load feature-engineered dataset
df = pd.read_csv(
    r"C:\Users\LENOVO\Desktop\project 1 infotech\project1_feature_engineered.csv"
)

# Clean column names for LightGBM
df.columns = [
    re.sub(r'[^A-Za-z0-9_]', '_', col)
    for col in df.columns
]

df.columns = [
    re.sub(r'_+', '_', col).strip('_')
    for col in df.columns
]

# Set target column
TARGET = "Machine_failure"

# Features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Convert categorical columns to numeric
X = pd.get_dummies(X, drop_first=True)

# Clean feature names again after one-hot encoding
X.columns = [
    re.sub(r'[^A-Za-z0-9_]', '_', col)
    for col in X.columns
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Original predictions
y_pred = model.predict(X_test)

# Add Gaussian noise
X_test_noisy = X_test.copy()

numeric_cols = X_test_noisy.select_dtypes(
    include=["int64", "float64"]
).columns

noise = np.random.normal(
    loc=0,
    scale=0.1,
    size=X_test_noisy[numeric_cols].shape
)

X_test_noisy[numeric_cols] += noise

# Predictions on noisy data
y_pred_noisy = model.predict(X_test_noisy)

# Compare results
print("Original Accuracy:", accuracy_score(y_test, y_pred))
print("Noisy Accuracy:", accuracy_score(y_test, y_pred_noisy))

print("Original F1-Score:", f1_score(y_test, y_pred))
print("Noisy F1-Score:", f1_score(y_test, y_pred_noisy))