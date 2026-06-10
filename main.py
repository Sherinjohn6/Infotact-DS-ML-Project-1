import pandas as pd

# Load CSV file
df = pd.read_csv("project1 infotech.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())