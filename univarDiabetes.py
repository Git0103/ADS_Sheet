import pandas as pd
from scipy.stats import skew, kurtosis

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Select only numerical columns (excluding Outcome which is binary)
numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Calculate statistics for each numerical column
stats_df = pd.DataFrame({
    'Mean': df[numerical_cols].mean(),
    'Median': df[numerical_cols].median(),
    'Variance': df[numerical_cols].var(),
    'Standard Deviation': df[numerical_cols].std(),
    'Skewness': df[numerical_cols].skew(),
    'Kurtosis': df[numerical_cols].apply(kurtosis)
})

# Add count of non-null values
stats_df['Count'] = df[numerical_cols].count()

# Reorder columns
stats_df = stats_df[['Count', 'Mean', 'Median', 'Variance', 
                    'Standard Deviation', 'Skewness', 'Kurtosis']]

# Display the results with 2 decimal places
print(stats_df.round(2))