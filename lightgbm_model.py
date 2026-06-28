
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# Load SMOTE dataset

df = pd.read_csv("smote_dataset.csv")
print(df.columns.tolist())

# Rename all columns to safe names
df.columns = [
    col.replace(" ", "_")
       .replace("[", "")
       .replace("]", "")
       .replace("(", "")
       .replace(")", "")
       .replace("/", "_")
    for col in df.columns
]

print(df.columns)
# Features and target
X = df.drop("Machine_failure", axis=1)
y = df["Machine_failure"]
print(df.columns.tolist())
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LightGBM model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from lightgbm import LGBMClassifier

# Train model
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Save model and data
import joblib

joblib.dump(model, "lightgbm_model.pkl")
joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_train, "y_train.pkl")
joblib.dump(y_test, "y_test.pkl")

print("All files saved successfully.")