import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                           r2_score, explained_variance_score, 
                           mean_absolute_percentage_error)

# Load the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', header=None, names=column_names)

# Split data
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and display multiple regression metrics"""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Adjusted R2': 1 - (1-r2_score(y_true, y_pred))*(len(y_true)-1)/(len(y_true)-X.shape[1]-1),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }
    
    print(f"\n{model_name} Performance Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

# Evaluate both models
lr_metrics = evaluate_model(y_test, y_pred_lr, "Linear Regression")
rf_metrics = evaluate_model(y_test, y_pred_rf, "Random Forest")