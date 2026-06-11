import pandas as pd

# Load the dataset
df = pd.read_csv("project1 infotech.csv")

# Display first 5 rows
print("Original Dataset:")
print(df.head())

# Feature Engineering

# 1. Temperature Difference
df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# 2. Mechanical Power
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

# 3. Wear Rate
df['Wear_Rate'] = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)

# 4. Temperature Ratio
df['Temp_Ratio'] = df['Process temperature [K]'] / df['Air temperature [K]']

# 5. Torque-RPM Ratio
df['Torque_RPM_Ratio'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1)

# 6. Encode Product Type
df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})

# 7. High Wear Indicator
df['High_Wear'] = (df['Tool wear [min]'] > 150).astype(int)

# 8. High Temperature Indicator
df['High_Temp'] = (df['Process temperature [K]'] > 310).astype(int)

# Save the updated dataset
df.to_csv("project1_feature_engineered.csv", index=False)

print("\nNew features added successfully!")
print("\nUpdated Dataset:")
print(df.head())

print("\nSaved as: project1_feature_engineered.csv")