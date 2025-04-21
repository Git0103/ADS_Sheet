import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer  # for imputation

# Load dataset
df = pd.read_csv('olympics.csv')

# Convert all applicable columns to numeric
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# Select numerical columns and drop rows where all are NaN
# The original code had a potential issue: If a non-numeric column
# contained only NaNs or empty strings after coercion, it might
# have been included in 'numerical_df', leading to a column count mismatch.
# The following line is changed to more accurately select only numeric columns.
numerical_df = df_numeric.select_dtypes(include=['number']).dropna(how='all', axis=1)

# Instead of dropping rows with NaNs, impute them using the mean
imputer = SimpleImputer(strategy='mean')  # or other strategy like 'median'

# Fit and transform on numerical_df, but keep all columns
# The column mismatch is addressed by using the columns from the imputed data
clean_df = pd.DataFrame(imputer.fit_transform(numerical_df),
                        columns=numerical_df.columns)  

# Record count before outlier removal
before_count = clean_df.shape[0]

# Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)  # 5% assumed outliers
y_pred = lof.fit_predict(clean_df)

# Keep only inliers (y_pred == 1)
cleaned_df = clean_df[y_pred == 1]

# Record count after outlier removal
after_count = cleaned_df.shape[0]

# Print counts
print("Data count BEFORE outlier detection:", before_count)
print("Data count AFTER outlier detection:", after_count)