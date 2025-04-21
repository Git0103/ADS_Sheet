import pandas as pd
import numpy as np

# Create synthetic shampoo sales data (36 months)
np.random.seed(42)
base_sales = np.linspace(100, 300, 36)
noise = np.random.normal(0, 20, 36)
sales = base_sales + noise
dates = pd.date_range(start='2019-01-01', periods=36, freq='M')

# Create DataFrame and save to CSV
shampoo_df = pd.DataFrame({'Date': dates, 'Sales': sales.round(2)})
shampoo_df.to_csv('shampoo-sales.csv', index=False)
print("Dataset saved as 'shampoo-sales.csv'")

# Load the dataset
shampoo_df = pd.read_csv('shampoo-sales.csv', parse_dates=['Date'])
shampoo_df.set_index('Date', inplace=True)

# Plot the time series
import matplotlib.pyplot as plt
shampoo_df.plot(figsize=(12, 6), title='Monthly Shampoo Sales')
plt.ylabel('Sales')
plt.show()

# Create lagged features
lags = 3  # Using 3 previous months to predict next month
for i in range(1, lags+1):
    shampoo_df[f'Lag_{i}'] = shampoo_df['Sales'].shift(i)

# Drop rows with NaN values (first 3 months)
shampoo_df.dropna(inplace=True)

# Split into features (X) and target (y)
X = shampoo_df.drop('Sales', axis=1)
y = shampoo_df['Sales']

# Split into train and test sets (last 6 months for testing)
split_point = len(shampoo_df) - 6
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Evaluate model performance
from sklearn.metrics import mean_squared_error

# Remove 'squared' parameter for older scikit-learn versions
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred)) 
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Print model coefficients
print("\nModel coefficients:")
for i, coef in enumerate(model.coef_):
    print(f"Lag {i+1}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.2f}")

# Function to forecast future periods
def forecast(model, last_known_values, periods):
    forecasts = []
    current_input = last_known_values.copy()
    
    for _ in range(periods):
        pred = model.predict([current_input])[0]
        forecasts.append(pred)
        # Update input for next period
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred
    
    return forecasts

# Get last known values (most recent lags)
last_values = shampoo_df.iloc[-1, 1:].values  # Exclude actual sales

# Forecast next 6 months
future_periods = 6
future_dates = pd.date_range(start=shampoo_df.index[-1] + pd.DateOffset(months=1), periods=future_periods, freq='M')
future_sales = forecast(model, last_values, future_periods)

# Plot future forecasts
plt.figure(figsize=(12, 6))
plt.plot(shampoo_df.index, shampoo_df['Sales'], label='Historical Sales')
plt.plot(future_dates, future_sales, 'r--', label='Forecasted Sales')
plt.title('Shampoo Sales Forecast')
plt.ylabel('Sales')
plt.legend()
plt.show()

print("\nFuture Sales Forecast:")
for date, sale in zip(future_dates, future_sales):
    print(f"{date.strftime('%Y-%m')}: {sale:.2f}")