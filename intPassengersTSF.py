import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the dataset, skipping header/footer
df = pd.read_csv(
    'international-airline-passengers.csv',
    #.csv file is incorrect here, remember to correct the file first
    skiprows=1,  # Skip header
    skipfooter=1,  # Skip footer
    engine='python',  # Required for skipfooter
    header=None,
    names=['Month', 'Passengers']
)

# Debug: Print first few rows to check the format
print("Raw data sample:")
print(df.head())

# Try parsing dates flexibly (handles "Jan 49", "Jan-49", "1949-01", etc.)
df['Month'] = pd.to_datetime(df['Month'], format='mixed')

# Debug: Check parsed dates
print("\nParsed dates sample:")
print(df.head())

# Drop rows where Month couldn't be parsed (if any)
df = df.dropna(subset=['Month'])

# Set index and enforce monthly frequency
df.set_index('Month', inplace=True)
df = df.asfreq('MS')  # 'MS' = Month Start frequency

# Debug: Check final DataFrame
print("\nFinal DataFrame:")
print(df.head())

# Seasonal decomposition (Multiplicative Model, period=12 for yearly seasonality)
try:
    result = seasonal_decompose(
        df['Passengers'], 
        model='multiplicative',
        period=12  # Explicitly set period for yearly seasonality
    )
    
    # Plot the decomposition
    plt.figure(figsize=(12, 8))
    result.plot()
    plt.suptitle('Seasonal Decomposition of Airline Passengers', fontsize=16)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error during decomposition: {e}")
    print(f"Number of observations: {len(df)}")
    print("DataFrame head:")
    print(df.head())