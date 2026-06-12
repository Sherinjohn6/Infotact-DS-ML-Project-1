import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load dataset
df = pd.read_csv("project1 infotech.csv")

# Drop columns that should not be used
df = df.drop(['UDI', 'Product ID'], axis=1)

# Convert categorical column to numeric
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Separate features and target
X = df.drop('Machine failure', axis=1)
y = df['Machine failure']

print("Before SMOTE:", Counter(y))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE:", Counter(y_resampled))
df_smote = pd.DataFrame(X_resampled, columns=X.columns)
df_smote['Machine failure'] = y_resampled

df_smote.to_csv("smote_dataset.csv", index=False)

print("SMOTE dataset saved successfully!")