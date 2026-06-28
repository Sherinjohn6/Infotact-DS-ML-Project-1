# ============================================
# Hyperparameter Tuning using LightGBM + SMOTE
# AI4I Predictive Maintenance Dataset
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# ============================================
# Load Dataset
# ============================================
df = pd.read_csv("project1_feature_engineered.csv")

# ============================================
# Encode Categorical Column
# ============================================
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])

# ============================================
# Features and Target
# ============================================
X = df.drop(["Machine failure", "UDI", "Product ID"], axis=1)
y = df["Machine failure"]

# ============================================
# Remove Special Characters from Column Names
# ============================================
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

print("Feature Names:")
print(X.columns.tolist())

# ============================================
# Train-Test Split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# Apply SMOTE
# ============================================
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

# ============================================
# LightGBM Model
# ============================================
lgbm = LGBMClassifier(random_state=42)

# ============================================
# Hyperparameter Grid
# ============================================
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [5, 10],
    "num_leaves": [31, 50],
    "subsample": [0.8, 1.0]
}

# ============================================
# Grid Search
# ============================================
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=2,
    error_score="raise"
)

print("\nStarting Hyperparameter Tuning...\n")

grid_search.fit(X_train_smote, y_train_smote)

# ============================================
# Best Parameters
# ============================================
print("\n========== BEST PARAMETERS ==========")
print(grid_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(grid_search.best_score_)

# ============================================
# Best Model
# ============================================
best_model = grid_search.best_estimator_

# ============================================
# Prediction
# ============================================
y_pred = best_model.predict(X_test)

# ============================================
# Evaluation
# ============================================
print("\n========== TEST RESULTS ==========")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))