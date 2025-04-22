import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv('/content/Automobile_data.csv')

# Replace '?' with NaN
df.replace('?', pd.NA, inplace=True)

# Convert appropriate columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# 1. Drop rows with any missing values
df_dropna = df.dropna()

# 2. Fill missing values with a constant
df_fill_constant = df.fillna(value='Unknown')

# 3. Fill missing values with mean/median/mode
df_fill_mean = df.copy()
for col in df_fill_mean.select_dtypes(include=['float64', 'int64']).columns:
    df_fill_mean[col].fillna(df_fill_mean[col].mean(), inplace=True)

df_fill_mode = df.copy()
for col in df_fill_mode.columns:
    df_fill_mode[col].fillna(df_fill_mode[col].mode()[0], inplace=True)

# 4. KNN Imputer for numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
knn_df = df[numeric_cols].copy()

# Convert to float to ensure compatibility with KNN
knn_df = knn_df.astype(float)

knn_imputer = KNNImputer(n_neighbors=3)
df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(knn_df), columns=knn_df.columns)

# Replace original numerical columns with KNN imputed ones
df_imputed = df.copy()
df_imputed[numeric_cols] = df_knn_imputed

print("Missing values handled using various imputation techniques.")
