import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('placement.csv')

# Function to count outliers using IQR method
def count_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), outliers

# Count outliers for each numerical column
cgpa_outliers_count, cgpa_outliers = count_outliers('cgpa')
marks_outliers_count, marks_outliers = count_outliers('placement_exam_marks')

# Print results
print("Outlier Analysis Results:")
print("\nCGPA:")
print(f"Number of outliers: {cgpa_outliers_count}")
print("Outlier values:")
print(cgpa_outliers['cgpa'].sort_values().unique())

print("\nPlacement Exam Marks:")
print(f"Number of outliers: {marks_outliers_count}")
print("Outlier values:")
print(marks_outliers['placement_exam_marks'].sort_values().unique())

# Create box plots
plt.figure(figsize=(12, 6))

# Box plot for CGPA
plt.subplot(1, 2, 1)
sns.boxplot(y=df['cgpa'], color='lightblue')
plt.title('Box Plot of CGPA')
plt.ylabel('CGPA')

# Annotate with outlier count
plt.text(0.05, 0.95, f'Outliers: {cgpa_outliers_count}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

# Box plot for Placement Exam Marks
plt.subplot(1, 2, 2)
sns.boxplot(y=df['placement_exam_marks'], color='lightgreen')
plt.title('Box Plot of Placement Exam Marks')
plt.ylabel('Marks')

# Annotate with outlier count
plt.text(0.05, 0.95, f'Outliers: {marks_outliers_count}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()