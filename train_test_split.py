import pandas as pd
from sklearn.model_selection import train_test_split

# Load feature-engineered dataset
df = pd.read_csv("project1_feature_engineered.csv")

# Features and target
X = df.drop(['Machine failure', 'UDI', 'Product ID'], axis=1)
y = df['Machine failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)