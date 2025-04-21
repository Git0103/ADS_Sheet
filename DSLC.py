#perform DSLC on any dataset
#perform in Notebook or Colab

#Step 1: Pproblem Definition
print("Step 1: Problem Definition")
print("Goal: Predict median housing prices based on 13 features including crime rate, number of rooms, age of homes, etc.")
print("This is a regression problem where we'll predict continuous home values.")

#Step 2: Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', header=None, names=column_names)

print("\nStep 2: Data Collection")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

#Step 3: Data cleaning
print("\nStep 3: Data Cleaning")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check for duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Check data types
print("\nData types:")
print(df.dtypes)

#Step 4: EDA
print("\nStep 4: Exploratory Data Analysis")

# Descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Target variable distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['MEDV'], bins=30, kde=True)
plt.title('Distribution of Median Home Values (MEDV)')
plt.xlabel('Median Value ($1000s)')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Key feature relationships
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.scatterplot(x='RM', y='MEDV', data=df, ax=axes[0])
axes[0].set_title('Rooms vs. Home Value')

sns.scatterplot(x='LSTAT', y='MEDV', data=df, ax=axes[1])
axes[1].set_title('Lower Status % vs. Home Value')

sns.scatterplot(x='PTRATIO', y='MEDV', data=df, ax=axes[2])
axes[2].set_title('Pupil-Teacher Ratio vs. Home Value')

plt.tight_layout()
plt.show()


#Step 5: Feature Engineering
print("\nStep 5: Feature Engineering")

# Create new features
df['ROOMSPERHOUSE'] = df['RM'] / df['AGE']
df['TAXPERROOM'] = df['TAX'] / df['RM']

# Log transform skewed features
df['LOGCRIM'] = np.log1p(df['CRIM'])
df['LOGLSTAT'] = np.log1p(df['LSTAT'])

# Check the new features
print("\nNew features created:")
print(df[['ROOMSPERHOUSE', 'TAXPERROOM', 'LOGCRIM', 'LOGLSTAT']].head())


#Step 6: Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("\nStep 6: Model Training")

# Prepare data
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nModels trained: Linear Regression and Random Forest")


#Step 7: Model evaluation
print("\nStep 7: Model Evaluation")

# Linear Regression metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nLinear Regression:")
print(f"MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")

print("\nRandom Forest:")
print(f"MSE: {mse_rf:.2f}, R2: {r2_rf:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_lr)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.title('Random Forest: Actual vs Predicted')
plt.xlabel('Actual MEDV')

plt.tight_layout()
plt.show()


#Step 8:Deployment
import joblib

print("\nStep 8: Deployment Simulation")

# Save the best model
joblib.dump(rf, 'housing_price_predictor.pkl')

# Create sample prediction function
def predict_price(model, features):
    """Predict housing price from input features"""
    prediction = model.predict([features])
    return prediction[0]

# Test with sample data
sample_features = [0.1, 25, 5, 0, 0.5, 6.5, 50, 4, 5, 300, 15, 390, 5, 0.13, 46.15, -2.3, 1.8]
pred = predict_price(rf, sample_features)

print(f"\nSample prediction for input features: ${pred*1000:.0f}")
print("\nModel saved as 'housing_price_predictor.pkl'")



