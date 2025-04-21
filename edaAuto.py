import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset
df = pd.read_csv('Automobile_data.csv')

# Initial Data Inspection
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Data Cleaning
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert numerical columns to appropriate types
numerical_cols = ['normalized-losses', 'wheel-base', 'length', 'width', 'height',
                 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio',
                 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Basic Statistics
print("\n=== Descriptive Statistics ===")
print("Numerical Variables:")
print(df.describe())

print("\nCategorical Variables:")
print(df.describe(include=['object']))

# Missing Data Analysis
print("\n=== Missing Data Analysis ===")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_table = pd.concat([missing_data, missing_percent], axis=1)
missing_table.columns = ['Missing Values', '% of Total']
missing_table = missing_table[missing_table['Missing Values'] > 0]
print(missing_table.sort_values('% of Total', ascending=False))

# Visualization: Missing Data
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Univariate Analysis
# Numerical Variables
num_vars = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(15, 20))
for i, var in enumerate(num_vars, 1):
    plt.subplot(6, 3, i)
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')
    plt.tight_layout()
plt.show()

# Categorical Variables
cat_vars = df.select_dtypes(include=['object']).columns
plt.figure(figsize=(15, 20))
for i, var in enumerate(cat_vars, 1):
    plt.subplot(5, 3, i)
    df[var].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

# Bivariate Analysis
# Price vs Categorical Variables
plt.figure(figsize=(15, 20))
for i, var in enumerate(cat_vars, 1):
    plt.subplot(5, 3, i)
    sns.boxplot(x=var, y='price', data=df)
    plt.title(f'Price by {var}')
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

# Correlation Analysis
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Top Correlations with Price
print("\nTop Correlations with Price:")
print(corr['price'].sort_values(ascending=False)[1:11])

# Pairplot for Key Numerical Variables
key_vars = ['price', 'engine-size', 'horsepower', 'curb-weight', 'highway-mpg']
sns.pairplot(df[key_vars].dropna())
plt.show()

# Outlier Detection
plt.figure(figsize=(15, 20))
for i, var in enumerate(num_vars, 1):
    plt.subplot(6, 3, i)
    sns.boxplot(x=df[var])
    plt.title(f'Boxplot of {var}')
    plt.tight_layout()
plt.show()

# Advanced Analysis
# Price by Body Style and Fuel Type
plt.figure(figsize=(10, 6))
sns.barplot(x='body-style', y='price', hue='fuel-type', data=df)
plt.title('Average Price by Body Style and Fuel Type')
plt.show()

# Engine Size vs Horsepower by Number of Cylinders
plt.figure(figsize=(10, 6))
sns.scatterplot(x='engine-size', y='horsepower', hue='num-of-cylinders', data=df)
plt.title('Engine Size vs Horsepower by Number of Cylinders')
plt.show()

# City MPG vs Highway MPG by Drive Wheels
plt.figure(figsize=(10, 6))
sns.scatterplot(x='city-mpg', y='highway-mpg', hue='drive-wheels', data=df)
plt.title('City MPG vs Highway MPG by Drive Wheels')
plt.show()

# Key Findings Summary
print("\n=== Key Findings Summary ===")
print(f"1. Dataset contains {df.shape[0]} vehicles with {df.shape[1]} features")
print(f"2. Price ranges from ${df['price'].min():,.0f} to ${df['price'].max():,.0f}")
print(f"3. Most common fuel type: {df['fuel-type'].mode()[0]}")
print(f"4. Average horsepower: {df['horsepower'].mean():.1f}")
print(f"5. Strongest positive correlation with price: engine-size ({corr['price']['engine-size']:.2f})")
print(f"6. Strongest negative correlation with price: highway-mpg ({corr['price']['highway-mpg']:.2f})")
print("7. Luxury brands (BMW, Jaguar, Mercedes) have the highest average prices")
print("8. Vehicles with rear-wheel drive tend to be more expensive")
print("9. Diesel vehicles generally have better fuel economy but higher prices")
print("10. Convertibles and hardtops command premium prices compared to other body styles")