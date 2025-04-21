#python code snippet for extracting data from any given csv file

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
file_path = 'your-dataset.csv'  # Replace with your actual dataset file
df = pd.read_csv(file_path)

# Basic Info
print("Dataset Info:\n")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# Preview
print("\nFirst 5 Rows:\n", df.head())

# Correlation Heatmap (for numerical data)
numeric_cols = df.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("\nNo numeric columns for correlation heatmap.")
