import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("smote_dataset.csv")

# Keep only numerical columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Create correlation matrix
corr_matrix = numeric_df.corr()

# Plot settings
plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

# Save heatmap image
plt.savefig("heatmap.png", dpi=300, bbox_inches="tight")

# Display heatmap
plt.show()