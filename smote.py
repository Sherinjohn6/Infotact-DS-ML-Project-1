from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

# Load the dataset
df = pd.read_csv("project1_feature_engineered.csv")
# Features and target
X = df.drop(['Machine failure', 'UDI', 'Product ID'], axis=1)
y = df['Machine failure']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Check class distribution before SMOTE
print("Before SMOTE:")
print(Counter(y_train))

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("\nAfter SMOTE:")
print(Counter(y_train_smote))

# Shapes
print("\nOriginal Training Shape:", X_train.shape)
print("SMOTE Training Shape:", X_train_smote.shape)
print("Testing Shape:", X_test.shape)