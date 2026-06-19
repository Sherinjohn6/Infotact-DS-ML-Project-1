import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("smote_dataset.csv")

# Define features and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate on original test data
y_pred = model.predict(X_test)

print("=== Original Test Data ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# Create noisy test data
X_test_noisy = X_test.copy()

# Select only numerical columns
numeric_cols = X_test_noisy.select_dtypes(include=np.number).columns

# Add Gaussian noise (5% of each feature's standard deviation)
noise_level = 0.05

for col in numeric_cols:
    std = X_test_noisy[col].std()
    noise = np.random.normal(
        loc=0,
        scale=noise_level * std,
        size=len(X_test_noisy)
    )
    X_test_noisy[col] += noise

# Evaluate on noisy data
y_pred_noisy = model.predict(X_test_noisy)

print("\n=== Noisy Test Data ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_noisy), 4))
print(classification_report(y_test, y_pred_noisy))