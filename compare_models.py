import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv("smote_dataset.csv")

# Clean column names
df.columns = [
    col.replace(" ", "_")
       .replace("[", "")
       .replace("]", "")
       .replace("(", "")
       .replace(")", "")
       .replace("/", "_")
    for col in df.columns
]

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

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

results = []

# Train and evaluate each model
for name, model in models.items():

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1 Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Training Time (s)": round(training_time, 4)
    })

# Display results
results_df = pd.DataFrame(results)

print("\nModel Comparison Results:")
print(results_df.sort_values(by="F1 Score", ascending=False))