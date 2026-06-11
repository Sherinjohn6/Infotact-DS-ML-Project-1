import pandas as pd

# Load CSV file
df = pd.read_csv("project1 infotech.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())
import pandas as pd

# Load CSV file
df = pd.read_csv("project1 infotech.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
print(list(df.columns))